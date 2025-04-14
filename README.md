# VectorDatabase

A **vector database** is a specialized type of database designed to store, manage, and search high-dimensional **vector embeddings**—numerical representations of complex data such as text, images, audio, or video. Unlike traditional databases (such as relational structure like SQL) that use exact matching or structured queries, vector databases use **similarity search algorithms** (like cosine similarity or Euclidean distance) to find data points that are "closest" in meaning or structure. These systems are essential in applications involving **machine learning**, **natural language processing**, and **artificial intelligence**, where data must be compared based on content or context rather than exact values. For example, in recommendation systems, a vector database can find products similar to a user’s preferences; in semantic search engines, it retrieves documents with meanings related to a query rather than exact keyword matches; and in computer vision, it helps match faces or objects by comparing feature vectors. Popular vector databases include **Pinecone**, **FAISS**, **Weaviate**, and **Milvus**, and they are increasingly used in AI-driven tools such as **chatbots**, **personal assistants**, and **generative AI systems** to deliver fast and intelligent responses based on contextual understanding.



### **Workflow Overview:**

1. **Tokenization and Vocabulary Building**  
   The process begins by tokenizing the input sentences (you can change it based on your textutal data) to extract individual words (tokens). A vocabulary is then created, containing all unique tokens across the dataset.

2. **Vectorization**  
   Each sentence is transformed into a vector based on the frequency of words it contains. These vectors are structured according to the size of the vocabulary, with each dimension representing a specific token.

3. **Storing in VectorStore**  
   The resulting sentence vectors are stored in a `VectorStore` instance, enabling efficient similarity retrieval later.

4. **Similarity Search**  
   When a new sentence (query) is provided, it is vectorized using the same vocabulary. Cosine similarity is then computed between the query vector and all stored vectors to find the most semantically similar sentences.

5. **Output Generation**  
   The system displays the original query along with the top matching sentences and their associated similarity scores, providing insight into the semantic relationships.


### **An example:**

In this project, a vector Database is built from scratch, for running it, first, download all file in a folder and second, run the MainCode file to import all other files and run the database. After running the model, you input your query (such as **"What is a neural network?"**) and database will retrieve close textual data to it and rank them based on the degree of similarity.


![Plot](https://github.com/user-attachments/assets/0b480c83-2b68-4532-9e56-a11f2fb3cb50)














