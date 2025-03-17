import faiss
print(faiss.__version__)
print(hasattr(faiss, "StandardGpuResources"))  # True가 나와야 정상