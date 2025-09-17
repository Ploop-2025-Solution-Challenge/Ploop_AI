import pinecone, os
pinecone.init(api_key="APIKEY", environment="us-east1-gcp")
# print(pinecone.list_indexes())  # 여기서 인덱스 목록이 보여야 정상
print(pinecone.list_indexes())
# index = pinecone.Index("member-embeddings")
# print(index.describe_index_stats())