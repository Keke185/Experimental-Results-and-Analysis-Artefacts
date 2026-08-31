"""
This micro-benchmark test is used to verify whether the classifier decision logic introduces a significant
delay on top of the preceding SBERT encoding stage.
"""
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

N_TRIALS = 200
N_WARMUP = 20

CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]


def build_features(sims):
    mean_sim = sims.mean(axis=1, keepdims=True)
    std_sim = sims.std(axis=1, keepdims=True)
    max_sim = sims.max(axis=1, keepdims=True)
    margin = max_sim - mean_sim
    return np.hstack([sims, mean_sim, std_sim, margin])


def old_threshold_decision(sim_vec, weak_th=0.30, aligned_th=0.50):
    score = sim_vec.max()
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    return "Mismatched"


def timeit(fn, n_trials, n_warmup):
    for i in range(n_warmup):
        fn()
    times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        fn()

        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0
    return times.mean(), times.std(), np.median(times)


def main():
    # Force the use of single-threaded CPU to suit resource-constrained edge scenarios.
    torch.set_num_threads(1)
    device = "cpu"
    print("*" * 70)
    print("  Microbenchmark: SBERT encoding vs decision-logic overhead (CPU only)")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    sample_text = df["item_text"].iloc[0]
    print(f"OK, Using a representative item_text (len={len(sample_text)} chars) for timing")

    print(f"\n Loading MiniLM on CPU")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True, device=device)

    print(" Precomputing capability vectors (this happens once, cloud-side, offline , not timed)")
    cap_vectors = model.encode(CAPABILITY_CHUNKS, normalize_embeddings=True)

    # Fit a small reference classifier on random-ish data just to time inference
    rng = np.random.RandomState(0)
    dummy_X = rng.randn(60, 9)
    dummy_y = rng.choice(["Aligned", "Weakly Aligned", "Mismatched"], size=60)
    scaler = StandardScaler().fit(dummy_X)
    clf = LogisticRegression(max_iter=200).fit(scaler.transform(dummy_X), dummy_y)

    def encode_step():
        model.encode([sample_text], normalize_embeddings=True, show_progress_bar=False)

    enc_mean, enc_std, enc_med = timeit(encode_step, N_TRIALS, N_WARMUP)
    print(f"\n 1.SBERT encoding (1 item, CPU, single-thread): "
          f"mean={enc_mean:.3f}ms  median={enc_med:.3f}ms  std={enc_std:.3f}ms")

    # First, calculate an embedding vector for the following decision logic steps
    item_vec = model.encode([sample_text], normalize_embeddings=True, show_progress_bar=False)[0]
    sim_vec = item_vec @ cap_vectors.T  # shape (6,)

    # Executing old decision-making logic
    def old_decision_step():
        old_threshold_decision(sim_vec)

    old_mean, old_std, old_med = timeit(old_decision_step, N_TRIALS, N_WARMUP)
    print(f"\n 2.OLD decision logic (max-similarity + threshold): "
          f"mean={old_mean:.5f}ms  median={old_med:.5f}ms  std={old_std:.5f}ms")

    # Implementing new decision-making logic: Features + Classifier
    def new_decision_step():
        feats = build_features(sim_vec.reshape(1, -1))
        feats_scaled = scaler.transform(feats)
        clf.predict(feats_scaled)

    new_mean, new_std, new_med = timeit(new_decision_step, N_TRIALS, N_WARMUP)
    print(f"\n 3.NEW decision logic (features + logistic regression classifier): "
          f"mean={new_mean:.5f}ms  median={new_med:.5f}ms  std={new_std:.5f}ms")

    # Print ratios
    print("\n" + "*" * 70)
    print(" " * 30 + "  SUMMARY")
    print("*" * 70)
    print(f"  SBERT encoding time:        {enc_mean:.3f} ms")
    print(f"  Old threshold decision:     {old_mean:.5f} ms  ({enc_mean / max(old_mean, 1e-9):,.0f}x cheaper than encoding)")
    print(f"  New classifier decision:    {new_mean:.5f} ms  ({enc_mean / max(new_mean, 1e-9):,.0f}x cheaper than encoding)")
    print(f"  Classifier vs threshold overhead: +{new_mean - old_mean:.5f} ms "
          f"({(new_mean - old_mean) / enc_mean * 100:.4f}% of one encoding call)")

if __name__ == "__main__":
    main()
