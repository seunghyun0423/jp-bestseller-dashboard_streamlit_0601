
import ast, json, os, re, time
from datetime import datetime
from collections import Counter
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

try:
    import google.generativeai as genai
except Exception:
    genai = None
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    build = None
    HttpError = Exception
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

st.set_page_config(page_title="JP Bestseller Dashboard", page_icon="📊", layout="wide")

DEFAULT_QUERIES = [
    "돈키호테 추천","돈키호테 쇼핑리스트","돈키호테 화장품 추천",
    "일본 쇼핑 추천","일본 드럭스토어 추천","일본 화장품 추천",
    "일본 뷰티템 추천","일본 선크림 추천","일본 파스 추천",
    "일본 안약 추천","일본 캐릭터 굿즈 추천","일본 산리오 굿즈",
    "일본 치이카와 굿즈","일본 포켓몬 굿즈"
]
BASE_STOPWORDS = [
    "일본","추천","쇼핑","돈키호테","브이로그","여행","구매","리뷰","하울",
    "가격","진짜","좋은","좋아요","입니다","그리고","제품","사용","영상",
    "오늘","이번","소개","제가","저는","너무","정말","그냥","하면","해서",
    "있는","없는","같아요","합니다","있습니다","여러분","http","https","www","com",
    "곤약젤리","곤약","젤리","간식","과자","킷캣","자가리코","도쿄바나나",
    "이치란","라멘","카레","푸딩","후리카케","로이스","우마이봉","편의점",
    "음료","맥주","사케","술"
]

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def clean_html(text):
    return re.sub("<.*?>", "", str(text))

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def to_csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def ensure_alias_list(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list): return parsed
        except Exception:
            pass
        if "," in x: return [v.strip() for v in x.split(",") if v.strip()]
        return [x]
    return [str(x)]

def make_candidate_df(video_df, top_n=500, min_len=2):
    if "clean_text" not in video_df.columns:
        cols = [c for c in ["title","description","tags","comments_text","transcript"] if c in video_df.columns]
        tmp = video_df.copy()
        if "tags" in tmp.columns:
            tmp["tags"] = tmp["tags"].apply(lambda x: " ".join(ensure_alias_list(x)))
        tmp["all_text"] = tmp[cols].fillna("").astype(str).agg(" ".join, axis=1)
        tmp["clean_text"] = tmp["all_text"].apply(clean_text)
        video_df["clean_text"] = tmp["clean_text"]
    words = " ".join(video_df["clean_text"].fillna("").astype(str)).split()
    candidate_words = [
        w for w in words
        if len(w) >= min_len and len(w) <= 25 and w not in BASE_STOPWORDS
        and not w.isdigit() and not re.fullmatch(r"[0-9a-zA-Z]+", w)
        and "http" not in w.lower() and "www" not in w.lower() and "com" not in w.lower()
    ]
    return pd.DataFrame(Counter(candidate_words).most_common(top_n), columns=["keyword","count"])

def search_youtube_videos(youtube, query, max_results=30, region_code="KR", order="relevance"):
    res = youtube.search().list(q=query, part="id", type="video", maxResults=max_results,
                                regionCode=region_code, relevanceLanguage="ko", order=order).execute()
    return [item["id"]["videoId"] for item in res.get("items", [])]

def get_video_details(youtube, video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        res = youtube.videos().list(part="snippet,statistics", id=",".join(batch)).execute()
        for item in res.get("items", []):
            sn, stt = item.get("snippet", {}), item.get("statistics", {})
            rows.append({
                "video_id": item.get("id",""),
                "title": sn.get("title",""),
                "description": sn.get("description",""),
                "tags": sn.get("tags",[]),
                "published_at": sn.get("publishedAt",""),
                "channel_title": sn.get("channelTitle",""),
                "view_count": int(stt.get("viewCount",0) or 0),
                "like_count": int(stt.get("likeCount",0) or 0),
                "comment_count": int(stt.get("commentCount",0) or 0),
            })
    return pd.DataFrame(rows)

def get_comments(youtube, video_id, max_comments=20):
    if max_comments <= 0: return ""
    comments = []
    try:
        res = youtube.commentThreads().list(part="snippet", videoId=video_id,
                                            maxResults=min(max_comments,100),
                                            textFormat="plainText", order="relevance").execute()
        for item in res.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"].get("textDisplay",""))
    except Exception:
        pass
    return " ".join(comments)

def get_transcript_text(video_id):
    if YouTubeTranscriptApi is None: return ""
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko","en","ja"])
        return " ".join([t.get("text","") for t in tr])
    except Exception:
        return ""

def parse_json_from_gemini(text):
    text = text.strip().replace("```json","").replace("```","").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match: text = match.group(0)
    return json.loads(text)

def classify_keywords_with_gemini(api_key, keywords):
    if genai is None: raise RuntimeError("google-generativeai가 설치되지 않았습니다.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
너는 일본 이커머스 MD이자 상품 데이터 분류 전문가야.
아래 키워드 중 한국 온라인몰에서 취급할 만한 일본 제품/브랜드/캐릭터 IP 후보만 골라 분류해.
식품/간식/음료/주류, 여행지, 일반 단어, 사람 이름, 숫자, URL 조각은 반드시 제외.
category는 화장품, 드럭스토어, 캐릭터/굿즈, 생활용품, 문구/잡화, 패션잡화 중 하나.
brand를 모르면 "", 브랜드명만 있으면 item_name은 "".
aliases에는 원본 키워드를 반드시 포함.
JSON 배열만 출력. 설명 금지.
출력 예:
[{{"category":"화장품","brand":"캔메이크","product_group":"색조","item_name":"","aliases":["캔메이크"]}}]
키워드 목록:
{keywords}
"""
    return parse_json_from_gemini(model.generate_content(prompt).text)

def build_product_rank(video_df, dictionary_df):
    if "clean_text" not in video_df.columns:
        video_df["clean_text"] = video_df.fillna("").astype(str).agg(" ".join, axis=1).apply(clean_text)
    mentions = []
    for _, row in video_df.iterrows():
        text = re.sub(r"\s+", " ", str(row.get("clean_text","")).lower())
        for item in dictionary_df.to_dict("records"):
            total, matched = 0, []
            for alias in ensure_alias_list(item.get("aliases", [])):
                alias_l = str(alias).lower().strip()
                if alias_l:
                    cnt = text.count(alias_l)
                    if cnt:
                        total += cnt
                        matched.append(alias)
            if total:
                mentions.append({
                    "category": item.get("category",""), "brand": item.get("brand",""),
                    "product_group": item.get("product_group",""), "item_name": item.get("item_name",""),
                    "matched_aliases": ", ".join(sorted(set(map(str, matched)))),
                    "mention_count_in_video": total, "video_id": row.get("video_id",""),
                    "title": row.get("title",""), "view_count": int(row.get("view_count",0) or 0),
                    "like_count": int(row.get("like_count",0) or 0),
                    "comment_count": int(row.get("comment_count",0) or 0),
                    "published_at": row.get("published_at",""), "channel_title": row.get("channel_title","")
                })
    mention_df = pd.DataFrame(mentions)
    if mention_df.empty: return mention_df, pd.DataFrame()
    rank = mention_df.groupby(["category","brand","product_group","item_name"], dropna=False).agg(
        mention_count=("mention_count_in_video","sum"), video_count=("video_id","nunique"),
        total_view_count=("view_count","sum"), avg_view_count=("view_count","mean"),
        total_like_count=("like_count","sum"), total_comment_count=("comment_count","sum"),
        matched_aliases=("matched_aliases", lambda x: ", ".join(sorted(set(", ".join(x).split(", ")))))
    ).reset_index()
    rank["view_score"] = (rank["total_view_count"] / rank["total_view_count"].max()) * 100
    rank["mention_score"] = (rank["mention_count"] / rank["mention_count"].max()) * 100
    rank["video_score"] = (rank["video_count"] / rank["video_count"].max()) * 100
    rank["sns_score"] = rank["mention_score"]*0.5 + rank["video_score"]*0.3 + rank["view_score"]*0.2
    rank["sns_rank"] = rank["sns_score"].rank(method="dense", ascending=False).astype(int)
    rank["view_rank"] = rank["total_view_count"].rank(method="dense", ascending=False).astype(int)
    rank["mention_rank"] = rank["mention_count"].rank(method="dense", ascending=False).astype(int)
    cols = ["sns_rank","mention_rank","view_rank","category","brand","product_group","item_name",
            "mention_count","video_count","total_view_count","avg_view_count","total_like_count",
            "total_comment_count","sns_score","matched_aliases"]
    return mention_df, rank[cols].sort_values("sns_rank")

def make_search_keyword(row):
    category = str(row.get("category","")).strip()
    brand = str(row.get("brand","")).strip()
    group = str(row.get("product_group","")).strip()
    item = str(row.get("item_name","")).strip()
    brand = "" if brand.lower() in ["nan","none"] else brand
    group = "" if group.lower() in ["nan","none"] else group
    item = "" if item.lower() in ["nan","none"] else item
    if brand and item: return f"{brand} {item}"
    if item: return f"일본 {item}"
    if category == "캐릭터/굿즈" and brand: return f"{brand} 굿즈"
    if brand and group: return f"{brand} {group}"
    if brand: return brand
    if group: return f"일본 {group}"
    return ""

def get_naver_shopping(client_id, client_secret, keyword, display=50, sort="sim"):
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {"query": keyword, "display": display, "start": 1, "sort": sort, "exclude": "used:rental"}
    res = requests.get(url, headers=headers, params=params, timeout=20)
    if res.status_code != 200:
        raise RuntimeError(f"{keyword} / status={res.status_code} / {res.text[:200]}")
    rows = []
    for item in res.json().get("items", []):
        rows.append({
            "search_keyword": keyword, "product_name": clean_html(item.get("title","")),
            "lowest_price": int(item.get("lprice",0) or 0), "highest_price": int(item.get("hprice",0) or 0),
            "mall_name": item.get("mallName",""), "brand_from_naver": item.get("brand",""),
            "maker": item.get("maker",""), "category1": item.get("category1",""),
            "category2": item.get("category2",""), "category3": item.get("category3",""),
            "category4": item.get("category4",""), "product_id": item.get("productId",""),
            "link": item.get("link",""), "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return pd.DataFrame(rows)

st.sidebar.title("📌 메뉴")
page = st.sidebar.radio("페이지 선택", ["1. 유튜브 크롤링", "2. Gemini 딕셔너리 랭킹", "3. 네이버 쇼핑 최저가 비교"])
st.sidebar.divider()
youtube_key = st.sidebar.text_input("YouTube Data API Key", value=get_secret("YOUTUBE_API_KEY",""), type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", value=get_secret("GEMINI_API_KEY",""), type="password")
naver_id = st.sidebar.text_input("Naver Client ID", value=get_secret("NAVER_CLIENT_ID",""), type="password")
naver_secret = st.sidebar.text_input("Naver Client Secret", value=get_secret("NAVER_CLIENT_SECRET",""), type="password")

if "queries" not in st.session_state:
    st.session_state.queries = DEFAULT_QUERIES.copy()

if page == "1. 유튜브 크롤링":
    st.title("📊 YouTube 키워드 분석 대시보드")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("1. 검색 키워드 관리")
        q = st.text_input("새 검색어 추가", placeholder="예: 일본 돈키호테 필수템")
        if st.button("검색어 추가") and q.strip():
            if q.strip() not in st.session_state.queries:
                st.session_state.queries.append(q.strip())
        rm = st.multiselect("삭제할 검색어 선택", st.session_state.queries)
        if st.button("선택한 검색어 삭제"):
            st.session_state.queries = [x for x in st.session_state.queries if x not in rm]
        st.dataframe(pd.DataFrame({"queries": st.session_state.queries}), use_container_width=True)
    with col2:
        st.subheader("2. 크롤링 옵션")
        c1,c2,c3 = st.columns(3)
        max_results = c1.slider("검색어별 영상 수", 10, 50, 30, 10)
        order_label = c2.selectbox("정렬 기준", ["관련도순","최신순","조회수순"])
        order = {"관련도순":"relevance","최신순":"date","조회수순":"viewCount"}[order_label]
        region = c3.selectbox("지역 코드", ["KR","JP","US"])
        collect_comments = st.checkbox("댓글 수집", True)
        max_comments = st.slider("영상별 댓글 수", 0, 100, 20, 10, disabled=not collect_comments)
        collect_transcript = st.checkbox("자막 수집", True)
        top_n = st.slider("후보 키워드 출력 개수", 100, 1000, 500, 100)
        st.info(f"예상 검색 호출 {len(st.session_state.queries)}회 / 영상 최대 {len(st.session_state.queries)*max_results:,}개")
        if st.button("🚀 크롤링 업데이트", type="primary"):
            if not youtube_key: st.error("YouTube API Key를 입력하세요.")
            else:
                try:
                    youtube = build("youtube","v3",developerKey=youtube_key)
                    ids = []
                    prog = st.progress(0); status = st.empty()
                    for i, query in enumerate(st.session_state.queries):
                        status.write(f"검색 중: {query}")
                        ids.extend(search_youtube_videos(youtube, query, max_results, region, order))
                        prog.progress((i+1)/len(st.session_state.queries)); time.sleep(0.2)
                    ids = list(dict.fromkeys(ids))
                    video_df = get_video_details(youtube, ids)
                    if collect_comments and max_comments:
                        video_df["comments_text"] = [get_comments(youtube, v, max_comments) for v in video_df["video_id"]]
                    else: video_df["comments_text"] = ""
                    if collect_transcript:
                        video_df["transcript"] = [get_transcript_text(v) for v in video_df["video_id"]]
                    else: video_df["transcript"] = ""
                    video_df["all_text"] = video_df["title"].fillna("")+" "+video_df["description"].fillna("")+" "+video_df["tags"].apply(lambda x: " ".join(x) if isinstance(x,list) else str(x))+" "+video_df["comments_text"].fillna("")+" "+video_df["transcript"].fillna("")
                    video_df["clean_text"] = video_df["all_text"].apply(clean_text)
                    st.session_state.video_df = video_df
                    st.session_state.candidate_df = make_candidate_df(video_df, top_n)
                except Exception as e:
                    st.error(f"오류: {e}")
    if "candidate_df" in st.session_state:
        st.divider()
        vdf, cdf = st.session_state.video_df, st.session_state.candidate_df
        a,b,c = st.columns(3)
        a.metric("영상 수", f"{len(vdf):,}")
        b.metric("후보 키워드", f"{len(cdf):,}")
        c.metric("자막 있는 영상", f"{(vdf.get('transcript','') != '').sum():,}")
        tab1, tab2 = st.tabs(["후보 키워드", "원본 영상"])
        with tab1:
            st.dataframe(cdf, use_container_width=True)
            fig = px.bar(cdf.head(30), x="count", y="keyword", orientation="h", title="상위 후보 키워드 TOP30")
            fig.update_layout(yaxis={"autorange":"reversed"}); st.plotly_chart(fig, use_container_width=True)
            st.download_button("candidate_df 다운로드", to_csv_bytes(cdf), "YT_candidate_df.csv", "text/csv")
        with tab2:
            st.dataframe(vdf, use_container_width=True)
            st.download_button("youtube_raw 다운로드", to_csv_bytes(vdf), "YT_video_raw.csv", "text/csv")

elif page == "2. Gemini 딕셔너리 랭킹":
    st.title("🧠 Gemini 딕셔너리 기반 인기 상품 랭킹")
    st.caption("1페이지에서 크롤링한 candidate_df와 raw 영상 데이터를 자동으로 불러옵니다. 필요할 때만 CSV를 직접 업로드하세요.")

    # --------------------------------------------------
    # 1) 1페이지 결과 자동 불러오기 + 수동 업로드 fallback
    # --------------------------------------------------
    candidate_df = st.session_state.get("candidate_df", None)
    video_df = st.session_state.get("video_df", None)

    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        if candidate_df is not None:
            st.success(f"candidate_df 자동 불러오기 완료: {len(candidate_df):,}행")
        else:
            st.warning("자동으로 불러올 candidate_df가 없습니다. CSV를 직접 업로드하세요.")
            candidate_file = st.file_uploader("candidate_df CSV 업로드", type=["csv"], key="cand_manual")
            if candidate_file is not None:
                candidate_df = pd.read_csv(candidate_file)
                st.session_state.candidate_df = candidate_df
                st.success(f"candidate_df 업로드 완료: {len(candidate_df):,}행")

    with col_auto2:
        if video_df is not None:
            st.success(f"YouTube raw 데이터 자동 불러오기 완료: {len(video_df):,}행")
        else:
            st.warning("자동으로 불러올 YouTube raw 데이터가 없습니다. CSV를 직접 업로드하세요.")
            raw_file = st.file_uploader("YouTube raw CSV 업로드", type=["csv"], key="raw_manual")
            if raw_file is not None:
                video_df = pd.read_csv(raw_file)
                st.session_state.video_df = video_df
                st.success(f"YouTube raw 업로드 완료: {len(video_df):,}행")

    st.divider()

    # --------------------------------------------------
    # 2) Gemini 옵션 + 수작업 딕셔너리 선택 업로드
    # --------------------------------------------------
    st.subheader("1. Gemini 딕셔너리 생성 옵션")
    c1, c2 = st.columns(2)
    min_count = c1.number_input("Gemini 분류 최소 빈도", 1, 100, 5)
    max_keywords = c2.slider("Gemini 분류 키워드 수", 50, 1000, 300, 50)

    manual_dict_file = st.file_uploader(
        "추가 수작업 딕셔너리 업로드 선택사항 CSV 또는 JSON",
        type=["csv", "json"],
        key="manual_dict_upload"
    )

    def load_manual_dictionary(uploaded_file):
        if uploaded_file is None:
            return pd.DataFrame(columns=["category", "brand", "product_group", "item_name", "aliases"])
        try:
            if uploaded_file.name.endswith(".csv"):
                df_manual = pd.read_csv(uploaded_file)
            else:
                data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                df_manual = pd.DataFrame(data)
            for col in ["category", "brand", "product_group", "item_name", "aliases"]:
                if col not in df_manual.columns:
                    df_manual[col] = ""
            df_manual["aliases"] = df_manual["aliases"].apply(ensure_alias_list)
            return df_manual[["category", "brand", "product_group", "item_name", "aliases"]]
        except Exception as e:
            st.error(f"수작업 딕셔너리 파일을 읽는 중 오류가 발생했습니다: {e}")
            return pd.DataFrame(columns=["category", "brand", "product_group", "item_name", "aliases"])

    manual_dict_df = load_manual_dictionary(manual_dict_file)
    if manual_dict_file is not None:
        st.info(f"수작업 딕셔너리 {len(manual_dict_df):,}개가 추가 대상으로 로드되었습니다.")

    if candidate_df is not None:
        st.subheader("2. candidate_df 미리보기")
        st.dataframe(candidate_df.head(100), use_container_width=True)

        if st.button("Gemini로 딕셔너리 생성", type="primary"):
            if not gemini_key:
                st.error("Gemini API Key를 입력하세요.")
            else:
                try:
                    kws = (
                        candidate_df[candidate_df["count"] >= min_count]
                        .sort_values("count", ascending=False)
                        .head(max_keywords)["keyword"]
                        .astype(str)
                        .tolist()
                    )
                    dictionary_df = pd.DataFrame(classify_keywords_with_gemini(gemini_key, kws))
                    for col in ["category", "brand", "product_group", "item_name", "aliases"]:
                        if col not in dictionary_df.columns:
                            dictionary_df[col] = ""
                    dictionary_df["aliases"] = dictionary_df["aliases"].apply(ensure_alias_list)
                    dictionary_df = dictionary_df.drop_duplicates(subset=["category", "brand", "product_group", "item_name"])

                    # 수작업 딕셔너리는 선택적으로만 결합
                    final_dictionary_df = pd.concat([dictionary_df, manual_dict_df], ignore_index=True)
                    final_dictionary_df = final_dictionary_df.drop_duplicates(
                        subset=["category", "brand", "product_group", "item_name"]
                    )

                    st.session_state.dictionary_df = dictionary_df
                    st.session_state.final_dictionary_df = final_dictionary_df
                    st.success(f"Gemini 딕셔너리 생성 완료: {len(dictionary_df):,}개")
                    if len(manual_dict_df) > 0:
                        st.success(f"수작업 딕셔너리 포함 최종 딕셔너리: {len(final_dictionary_df):,}개")
                except Exception as e:
                    st.error(f"Gemini 오류: {e}")
    else:
        st.info("candidate_df가 있어야 Gemini 딕셔너리를 만들 수 있습니다. 1페이지에서 크롤링하거나 CSV를 업로드하세요.")

    # --------------------------------------------------
    # 3) 딕셔너리 확인 및 product_rank 생성
    # --------------------------------------------------
    if "final_dictionary_df" in st.session_state:
        st.divider()
        st.subheader("3. 최종 딕셔너리")
        ddf = st.session_state.final_dictionary_df
        st.dataframe(ddf, use_container_width=True)
        st.download_button("최종 딕셔너리 다운로드", to_csv_bytes(ddf), "YT_gemini_dictionary_final.csv", "text/csv")

        if video_df is not None:
            if st.button("딕셔너리 기반 product_rank 생성", type="primary"):
                mention_df, product_rank = build_product_rank(video_df, ddf)
                st.session_state.mention_df = mention_df
                st.session_state.product_rank = product_rank
                st.session_state.product_rank_df = product_rank
                st.success("product_rank가 생성되었습니다. 3페이지 네이버 쇼핑 최저가 비교에서 자동으로 사용할 수 있습니다.")
        else:
            st.warning("product_rank를 만들려면 YouTube raw 데이터가 필요합니다. 1페이지에서 크롤링하거나 raw CSV를 업로드하세요.")

    if "product_rank" in st.session_state:
        pr = st.session_state.product_rank
        st.subheader("인기 상품 랭킹")
        st.dataframe(pr, use_container_width=True)
        if not pr.empty:
            fig = px.bar(
                pr.head(30),
                x="sns_score",
                y="brand",
                color="category",
                orientation="h",
                title="SNS 점수 기준 TOP30"
            )
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)
        if "mention_df" in st.session_state:
            st.download_button("mention_df 다운로드", to_csv_bytes(st.session_state.mention_df), "YT_mention_df_gemini.csv", "text/csv")
        st.download_button("product_rank 다운로드", to_csv_bytes(pr), "YT_product_rank_gemini.csv", "text/csv")

elif page == "3. 네이버 쇼핑 최저가 비교":
    st.title("🛒 네이버 쇼핑 최저가 비교")
    st.caption("2페이지에서 생성한 product_rank를 자동으로 불러옵니다. 필요할 때만 CSV를 직접 업로드하세요.")

    product_rank = st.session_state.get("product_rank", None)
    if product_rank is None:
        product_rank = st.session_state.get("product_rank_df", None)

    if product_rank is not None:
        st.success(f"Gemini 랭킹 페이지에서 생성된 product_rank를 자동으로 불러왔습니다: {len(product_rank):,}행")
    else:
        st.warning("자동으로 불러올 product_rank가 없습니다. CSV를 직접 업로드하세요.")
        rank_file = st.file_uploader("product_rank CSV 업로드", type=["csv"], key="rank_manual")
        if rank_file is not None:
            product_rank = pd.read_csv(rank_file)
            st.session_state.product_rank = product_rank
            st.success(f"product_rank 업로드 완료: {len(product_rank):,}행")

    if product_rank is not None:
        c1, c2, c3 = st.columns(3)
        top_n = c1.slider("랭킹 상위 N개 검색", 10, 300, 100, 10)
        display_n = c2.slider("검색어별 상품 수", 10, 100, 50, 10)
        sort = {"정확도순":"sim", "최저가순":"asc", "최고가순":"dsc", "날짜순":"date"}[c3.selectbox("네이버 정렬", ["정확도순", "최저가순", "최고가순", "날짜순"])]

        target = product_rank.sort_values("sns_rank").head(top_n).copy()
        target["search_keyword"] = target.apply(make_search_keyword, axis=1)
        target = target[target["search_keyword"].astype(str).str.strip() != ""].drop_duplicates("search_keyword")

        st.info(f"최종 검색어 수 {len(target):,}개 / 예상 상품 수 최대 {len(target) * display_n:,}개")
        st.dataframe(target[["sns_rank", "category", "brand", "product_group", "item_name", "search_keyword"]], use_container_width=True)

        if st.button("네이버 쇼핑 수집 실행", type="primary"):
            if not naver_id or not naver_secret:
                st.error("Naver Client ID / Secret을 입력하세요.")
            else:
                dfs = []
                prog = st.progress(0)
                for i, (_, row) in enumerate(target.iterrows()):
                    try:
                        temp = get_naver_shopping(naver_id, naver_secret, row["search_keyword"], display_n, sort)
                        if not temp.empty:
                            for src, dst in [("sns_rank", "sns_rank"), ("mention_rank", "mention_rank"), ("view_rank", "view_rank")]:
                                temp[dst] = row.get(src, "")
                            temp["yt_category"] = row.get("category", "")
                            temp["yt_brand"] = row.get("brand", "")
                            temp["yt_product_group"] = row.get("product_group", "")
                            temp["yt_item_name"] = row.get("item_name", "")
                            temp["yt_mention_count"] = row.get("mention_count", "")
                            temp["yt_video_count"] = row.get("video_count", "")
                            temp["yt_total_view_count"] = row.get("total_view_count", "")
                            temp["yt_sns_score"] = row.get("sns_score", "")
                            dfs.append(temp)
                    except Exception as e:
                        st.warning(f"{row['search_keyword']} 실패: {e}")
                    prog.progress((i + 1) / len(target))
                    time.sleep(0.15)

                if dfs:
                    naver_df = pd.concat(dfs, ignore_index=True)
                    naver_df = naver_df[naver_df["lowest_price"] > 0].drop_duplicates(subset=["search_keyword", "product_id"])
                    naver_lowest = naver_df.sort_values(["search_keyword", "lowest_price"]).groupby("search_keyword").head(10).reset_index(drop=True)
                    naver_summary = naver_df.groupby(
                        ["sns_rank", "yt_category", "yt_brand", "yt_product_group", "yt_item_name", "search_keyword"],
                        dropna=False
                    ).agg(
                        naver_product_count=("product_id", "nunique"),
                        min_price=("lowest_price", "min"),
                        avg_price=("lowest_price", "mean"),
                        median_price=("lowest_price", "median"),
                        max_price=("lowest_price", "max"),
                        mall_count=("mall_name", "nunique"),
                        yt_mention_count=("yt_mention_count", "first"),
                        yt_video_count=("yt_video_count", "first"),
                        yt_total_view_count=("yt_total_view_count", "first"),
                        yt_sns_score=("yt_sns_score", "first")
                    ).reset_index().sort_values("sns_rank")

                    st.session_state.naver_df = naver_df
                    st.session_state.naver_lowest = naver_lowest
                    st.session_state.naver_summary = naver_summary
                    st.success("네이버 쇼핑 수집이 완료되었습니다.")

    if "naver_summary" in st.session_state:
        st.subheader("네이버 쇼핑 요약")
        st.dataframe(st.session_state.naver_summary, use_container_width=True)
        fig = px.scatter(
            st.session_state.naver_summary,
            x="yt_sns_score",
            y="min_price",
            size="naver_product_count",
            color="yt_category",
            hover_name="search_keyword",
            title="YouTube 인기 점수 vs 네이버 최저가"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("raw 전체 다운로드", to_csv_bytes(st.session_state.naver_df), "naver_shopping_raw_from_yt_rank.csv", "text/csv")
        st.download_button("최저가 TOP10 다운로드", to_csv_bytes(st.session_state.naver_lowest), "naver_shopping_lowest_top10_from_yt_rank.csv", "text/csv")
        st.download_button("요약표 다운로드", to_csv_bytes(st.session_state.naver_summary), "naver_shopping_summary_from_yt_rank.csv", "text/csv")

