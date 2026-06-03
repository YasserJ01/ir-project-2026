# Practical Project: Information Retrieval Systems 2026

**Project Title:** Building an Information Retrieval System  
**Theoretical Course Professor:** Dr. Abi Sandouk  
**Practical Lab Instructors:** Eng. Marwa Al-Daya, Eng. Salyma Al-Muhairi  
**Final Submission Deadline:** July 3rd (7/3)

---

## Project Description
You are required to build a custom search engine that can search and retrieve documents from two different datasets. Information retrieval principles must be used to design and implement a search engine capable of handling user queries and returning relevant text results in natural language.

Each group must choose **two different datasets** from the following link: [https://ir-datasets.com](https://ir-datasets.com), provided that **each individual dataset contains more than 200K documents**. Additionally, the datasets must contain **testing data** and **qrel** (i.e., query examples and the ground-truth relevance judgements/most suitable results for these queries).

---

## System Requirements & Verification
The group must build and verify an information retrieval system that contains the following:

### 1. Data Pre-Processing
After downloading and loading the data, it must be processed according to what the group deems appropriate (e.g., Stemming, Lemmatization, Normalization, etc.).

### 2. Document Representation
Documents in each dataset must be represented using **all** of the following methods:
*   **VSM_TF-IDF Representation**
*   **Embedding Representation** (such as Word2Vec, BERT, etc.)
*   **BM25 Representation**
*   **Hybrid Representation** (Serial or Parallel)

#### Important Notes on Representation:
1. When using the **Hybrid Representation in a parallel manner**, result merging methods (**Fusion Methods**) must be employed to calculate the final scores (**Scoring**) for document ranking.
2. For **BM25**, a method must be provided to observe the parameter changes based on the query during execution within the User Interface, or a clear justification must be provided in the report explaining why specific parameter values were adopted.
3. The **Hybrid Representation must be applied twice**: once in a parallel (divergent) layout and once in a serial layout. The User Interface must provide a clear option allowing the user to search using either the Serial Hybrid Representation or the Parallel Hybrid Representation.
4. Multiple types of Embedding models can be combined and used together within the Parallel Hybrid Representation.

### 3. Indexing
Build one or more indexes suitable for each dataset (such as an **Inverted Index**, etc.) to retrieve documents quickly and effectively. Indexing terms must be chosen with high efficiency.

### 4. Query Processing
Queries must be processed using the exact same pre-processing techniques and represented using the same document representation methods to ensure full compatibility between the queries and the retrieved documents.

### 5. Query Refinement
Apply enhancements to queries to increase the accuracy of the results. This includes techniques such as:
*   Weighting the user's query with information extracted from their previous search history (Personalization/Logs).
*   Adding synonyms to the query terms.
*   Linguistically/Grammatically correcting the query, etc.

### 6. Query Matching & Ranking
Build a function to match the query representation with the documents and rank the results according to the highest similarity scores. The appropriate matching method must be adopted for each specific representation model (e.g., VSM | Embedding $ightarrow$ Cosine Similarity, etc.).

### 7. Application of Service-Oriented Architecture (SOA)
The system must be designed following the **Service-Oriented Architecture (SOA)** concept, where the system is divided into a collection of independent services. Each part must be responsible for a specific task and be capable of operating, deploying, and developing separately.

For example, the system can be decoupled into the following services:
*   **Preprocessing Service**
*   **Indexing Service**
*   **Retrieval Service**
*   **Ranking & Evaluation Service**
*   **Query Refinement Service**
*   **Frontend Service or API Gateway**

#### Architecture Considerations:
*   Achieve a clear separation of responsibilities/concerns between services.
*   Adopt an appropriate communication protocol/style between services (e.g., REST API, Message Queue, RPC, etc.).
*   Write clean, organized, maintainable, and extensible code.
*   Ensure the ability to run or test each Service independently.
*   Apply best practices and appropriate Design Patterns to optimize performance and scalability.
*   Illustrate the system architecture and the communication workflow between services within the report using an **Architecture Diagram**.

*Note: The grade for this section increases based on how organized, flexible, and professional the service design is regarding:*
*   Clean Architecture
*   Scalability
*   Maintainability
*   Loose Coupling
*   Reusability

It is highly preferred to explain the rationale behind choosing the specific architecture and technologies used, detailing how they contributed to improving system performance or facilitating future development and scalability.

### 8. System Evaluation
The performance of the information retrieval system must be evaluated using standard benchmark metrics approved in the field of IR to verify the quality of the retrieved results and their relevance to the queries.

The following metrics must be calculated at a minimum for **each representation model** and for **each Dataset**:
*   **Mean Average Precision (MAP)**
*   **Recall**
*   **Precision@10**
*   **Normalized Discounted Cumulative Gain (nDCG)**

#### Evaluation Scenarios:
The evaluation must be performed under two conditions:
1. **Before** applying the additional features.
2. **After** applying the additional features.

The report must provide a clear analysis of the results explaining:
*   The impact of each individual representation method on retrieval quality.
*   A comparative performance analysis of the different models (TF-IDF, BM25, Embeddings, Hybrid).
*   The degree to which additional features contribute to improving results or retrieval speed.
*   Justifications for the chosen models and parameters based on practical empirical results.

*Crucial Note: Any system that yields extremely low or illogical evaluation results compared to the nature of the utilized dataset will be strictly rejected.*

### 9. User Interface (UI Web or Mobile Application)
Build a user-friendly interface that includes:
*   Dataset selection from the interface prior to executing a search query.
*   Accepting user text queries.
*   Displaying relevant results retrieved from the selected dataset.
*   An option to execute the search using **basic requirements only**, or another option to execute using **basic requirements combined with additional features**.
*   The capability to interactively control the parameters of the probabilistic model (BM25 coefficients) directly through the interface.
*   The capability to choose the specific Hybrid Representation configuration through the UI.

---

## Additional System Features
Depending on the group size, additional advanced features must be implemented from the following list:
10. **RAG (Retrieval-Augmented Generation)**
11. **Use Vector Stores**
12. **Multilingual Retrieval System**
13. **Crawling**
14. **Distributed Information Retrieval**
15. **Documents Clustering**
16. **Personalization**
17. **Topic Detection**
18. **Agents**
19. **LTR (Learning to Rank)**

*Note: Groups can propose an alternative additional feature, provided it is explicitly approved by the practical lab instructor or the course professor.*

---

## Submission Requirements
The final deliverables must include:
1.  **Detailed Report in Arabic** describing the design and implementation of the information retrieval system, fully cited with references.
2.  **Description of the used datasets** (the system must be built using 2 datasets from the recommendations, not just one).
3.  **Detailed breakdown of the project steps** and descriptions for each service within the system.
4.  **System Architecture description**, clarifying the infrastructure of the services and how they communicate with each other according to the SOA concept outlined above.
5.  **Evaluation Reports** as specified in the evaluation section.
6.  **Work distribution/division** among the group members.
7.  **An executable version** of both search engines, completely ready to handle real-time queries during the project defense/interview.
8.  **GitHub Repository Link** containing the source code, featuring a clear and descriptive `README.md` explaining the code structure.

---

## Organizational & Administrative Notes
*   **Programming Language:** The project must be implemented **exclusively in Python**.
*   **Group Size & Feature Commitments:** Groups can consist of **5 to 7 members**, scaling requirements as follows:
    *   If the group has **5 students**: **1 additional feature** must be implemented.
    *   If the group has **6 students**: **2 additional features** must be implemented.
    *   If the group has **7 students**: **3 additional features** must be implemented.
*   **Dataset Substitution:** One of the proposed datasets from `ir-datasets.com` may be replaced with an external dataset, subject to prior approval from the lab instructor.
*   **Prohibited Datasets:** Usage of the **Antique dataset is strictly prohibited**.
