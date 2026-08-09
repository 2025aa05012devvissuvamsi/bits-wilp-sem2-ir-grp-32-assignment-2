"""
Crawling interface (assignment requirement B).

Re-crawling changes the document set, which makes the index/search/
recommender/evaluation results stale. To avoid ever leaving the app in an
inconsistent state, "Run Crawl" always cascades into an index rebuild in
the same action. Ingesting a secondary dataset does the same.
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from crawler.wiki_crawler import DEFAULT_SEEDS, WikiCrawler
from utils.streamlit_helpers import clear_all_caches, crawl_data_exists, get_metadata

st.set_page_config(page_title="Crawler", layout="wide")
st.title("Crawling Interface")

st.markdown(
    "Crawls Wikipedia breadth-first from multiple seed topics, round-robin across seeds "
    "so no single seed's link count can starve the others, with duplicate URL/document "
    "handling and retry+backoff on rate limiting."
)

if crawl_data_exists():
    meta = get_metadata()
    st.info(f"Current corpus: {len(meta)} pages processed, {int((~meta['is_duplicate']).sum())} stored.")

st.warning(
    "Running a new crawl replaces the current document set. The index, search engine, "
    "recommender, and evaluation results all depend on it, so this action will "
    "**automatically rebuild the index afterward** to keep everything consistent."
)

with st.form("crawl_config"):
    seeds_text = st.text_area(
        "Seed topics (one per line)", value="\n".join(DEFAULT_SEEDS), height=160,
    )
    c1, c2, c3 = st.columns(3)
    max_depth = c1.number_input("Max crawl depth", min_value=0, max_value=3, value=1)
    max_pages = c2.number_input("Max pages", min_value=5, max_value=500, value=50, step=5)
    near_dup_threshold = c3.slider("Near-duplicate Jaccard threshold", 0.5, 1.0, 0.85, 0.01)
    submitted = st.form_submit_button("Run Crawl + Rebuild Index", type="primary")

if submitted:
    seeds = [s.strip() for s in seeds_text.splitlines() if s.strip()]
    if not seeds:
        st.error("Provide at least one seed topic.")
        st.stop()

    progress_bar = st.progress(0.0, text="Starting crawl...")
    log_placeholder = st.empty()
    log_lines = []

    def progress_callback(msg: str):
        log_lines.append(msg)
        log_placeholder.code("\n".join(log_lines[-15:]))
        match = re.match(r"\[(\d+)/(\d+)\]", msg)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            progress_bar.progress(min(current / total, 1.0), text=f"Crawling... {current}/{total}")

    crawler = WikiCrawler(seeds=seeds, max_depth=int(max_depth), max_pages=int(max_pages),
                           near_dup_threshold=near_dup_threshold)
    result = crawler.crawl(progress_callback=progress_callback)
    progress_bar.progress(1.0, text="Crawl complete.")
    st.success(
        f"Crawl summary: {result['total_processed']} processed, "
        f"{result['stored']} stored, {result['duplicates']} duplicates."
    )

    with st.spinner("Rebuilding index (inverted index, vector index, link graph)..."):
        from indexing.index_manager import build_all
        stats = build_all()
    st.success(f"Index rebuilt: {stats['vocabulary_size']} terms, {stats['graph_edges']} link-graph edges.")

    clear_all_caches()
    st.info("Caches cleared. Other pages will now reflect the new corpus.")

st.divider()
st.subheader("Add a secondary dataset source (optional)")
st.caption(
    "Upload a CSV with a text column and a title column to merge an external "
    "dataset into the corpus alongside the Wikipedia crawl (heterogeneous sources)."
)
SAMPLE_DATASET_CSV = """title,text
Cloud Computing,"Cloud computing refers to the on-demand delivery of computing resources such as servers, storage, databases, and software over the internet. Instead of owning physical hardware, organizations rent capacity from providers and pay only for what they use. Common service models include Infrastructure as a Service, Platform as a Service, and Software as a Service, each offering a different balance of control and convenience. Cloud platforms enable rapid scaling, reduce upfront infrastructure costs, and support distributed teams working from anywhere. Key concerns include data security, vendor lock-in, and managing costs as usage grows."
Blockchain Technology,"A blockchain is a distributed digital ledger that records transactions across many computers so that no single record can be altered without changing all subsequent blocks and gaining agreement from the network. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data, forming a tamper-resistant chain. Blockchains underpin cryptocurrencies but are also explored for supply chain tracking, digital identity, and smart contracts. Public blockchains are open to anyone, while private or permissioned blockchains restrict participation to approved members. Scalability and energy consumption remain active areas of research and debate."
Cybersecurity Fundamentals,"Cybersecurity is the practice of protecting systems, networks, and data from unauthorized access, damage, or disruption. Core principles include confidentiality, integrity, and availability, often referred to as the CIA triad. Common threats include malware, phishing, ransomware, and denial-of-service attacks. Defensive measures range from firewalls and encryption to multi-factor authentication and regular security audits. As organizations increasingly rely on interconnected systems, cybersecurity has become a shared responsibility spanning technical controls, employee awareness, and organizational policy."
Big Data Analytics,"Big data refers to datasets that are too large or complex for traditional data-processing software to handle efficiently. It is often characterized by the three Vs: volume, velocity, and variety. Big data analytics uses techniques such as distributed computing, machine learning, and statistical modeling to extract patterns and insights from these datasets. Frameworks like Hadoop and Spark allow processing to be distributed across clusters of machines. Applications span recommendation systems, fraud detection, and predictive maintenance, though data quality and privacy remain persistent challenges."
Internet of Things,"The Internet of Things describes a network of physical devices embedded with sensors, software, and connectivity that allows them to collect and exchange data. Examples range from smart thermostats and wearable fitness trackers to industrial sensors monitoring factory equipment. IoT systems typically involve a device layer for data collection, a network layer for transmission, and a processing layer for analysis and decision-making. Benefits include improved efficiency and real-time monitoring, while challenges include device security, interoperability between manufacturers, and managing the sheer volume of generated data."
Quantum Computing Basics,"Quantum computing is a computing paradigm that uses principles of quantum mechanics, such as superposition and entanglement, to process information in fundamentally different ways than classical computers. While classical bits represent either 0 or 1, quantum bits, or qubits, can represent combinations of both states simultaneously. This allows certain problems, such as factoring large numbers or simulating molecular interactions, to potentially be solved much faster than with classical approaches. Quantum computers remain largely experimental, facing challenges around qubit stability, error correction, and scaling to practical sizes."
Edge Computing,"Edge computing moves data processing closer to where data is generated, such as on local devices or nearby servers, rather than relying solely on centralized cloud data centers. This reduces latency, which is important for applications like autonomous vehicles, industrial automation, and augmented reality that require near-instant responses. Edge computing also reduces the amount of raw data that must be transmitted over networks, easing bandwidth demands. It is often used alongside cloud computing in a hybrid architecture, with the edge handling time-sensitive tasks and the cloud handling large-scale storage and analysis."
DevOps Practices,"DevOps is a set of practices that combines software development and IT operations to shorten development cycles and deliver reliable software continuously. It emphasizes automation, particularly through continuous integration and continuous deployment pipelines, which allow code changes to be tested and released frequently and safely. DevOps culture encourages close collaboration between developers, operations staff, and quality assurance teams, breaking down traditional silos. Common tools include version control systems, containerization platforms, and infrastructure-as-code frameworks that let teams define and manage infrastructure using code."
Data Privacy Regulation,"Data privacy regulation governs how organizations collect, store, process, and share personal information about individuals. Laws such as the General Data Protection Regulation in Europe grant individuals rights over their data, including access, correction, and deletion, while imposing obligations on organizations to justify data collection and secure it appropriately. Non-compliance can result in significant financial penalties. Data privacy has become closely linked with cybersecurity practices, since protecting personal data from breaches is often a legal requirement as well as a security concern."
5G Wireless Networks,"5G is the fifth generation of mobile network technology, designed to offer significantly higher data speeds, lower latency, and greater device density compared to previous generations. It enables new use cases such as real-time remote control of machinery, high-definition video streaming on the move, and large-scale IoT deployments in smart cities. 5G networks use higher frequency radio waves, which offer more bandwidth but shorter range, requiring a denser network of smaller cell stations. Rollout has varied globally due to differences in infrastructure investment and spectrum availability."
Containerization and Virtualization,"Containerization packages an application together with its dependencies into a single lightweight unit that can run consistently across different computing environments. Unlike traditional virtual machines, which each include a full operating system, containers share the host system's kernel, making them faster to start and more resource-efficient. Popular tools include Docker for building containers and Kubernetes for orchestrating them at scale across clusters of machines. This approach has become central to modern software deployment, particularly in cloud-native and microservices architectures."
API Design Principles,"An application programming interface, or API, defines how different software components communicate with one another. Well-designed APIs are consistent, predictable, and well-documented, making it easier for developers to integrate systems without needing to understand their internal implementation. REST is a widely used architectural style for web APIs, relying on standard HTTP methods and stateless requests. Good API design also considers versioning, so that changes do not break existing integrations, and security measures such as authentication tokens to control access to sensitive operations."
"""

st.download_button(
    "Download sample dataset CSV",
    data=SAMPLE_DATASET_CSV,
    file_name="sample_dataset.csv",
    mime="text/csv",
    help="12 original short articles on tech topics not covered by the Wikipedia crawl seeds — download, then upload it below to test/demonstrate the heterogeneous-source ingestion.",
)
uploaded = st.file_uploader("Upload a CSV with text documents", type="csv")
if uploaded:
    text_col = st.text_input("Text column name", value="text")
    title_col = st.text_input("Title column name", value="title")
    if st.button("Ingest dataset"):
        os.makedirs("data", exist_ok=True)
        tmp_path = "data/_tmp_upload.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        from crawler.dataset_loader import load_dataset
        try:
            n_added = load_dataset(tmp_path, text_col=text_col, title_col=title_col)
        except KeyError as e:
            st.error(f"Column not found in uploaded CSV: {e}")
            st.stop()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        with st.spinner("Rebuilding index to include the new documents..."):
            from indexing.index_manager import build_all
            build_all()

        clear_all_caches()
        st.success(f"Ingested {n_added} documents and rebuilt the index — new docs are now searchable.")