# Collocation Classification with Unsupervised Relation Vectors #

Repository for the ACL 2019 paper _Collocation Classification with Unsupervised Relation Vectors_ [1]. It contains:

- Source code to run the experiments described in the paper. Run the following command from the root directory. The different vector operations and datasets can be changed in `src/config.py`:

```bash
        - python src/run.py
```

- The newly released __LexFunc__ dataset (in `data/collocations`), more than 5k collocations categorized according to their corresponding _lexical function_ [2]. The LexFunc dataset is continuously growing, the latest version is always kept [here](https://www.upf.edu/web/taln/english-collocations). 

- SeVeN [3] pretrained relation vectors for frequently co-occurring DiffVec relations and collocations. You may download them from [here](https://drive.google.com/drive/folders/1MAkqrtEP2wYVtUHfd4XU5EQHZ8Jei6sM?usp=sharing).

***

[1] _Espinosa-Anke, L., Wanner, L. and Schockaert, S. Collocation Classification with Unsupervised Relation Vectors. In Proceedings of the 57th Meeting of the Association for Computational Linguistics. Short papers. (2019)._

[2] _Mel’čuk, I. (1998). Collocations and lexical functions. Phraseology. Theory, analysis, and applications, 23-53._

[3] _Espinosa-Anke, L. and Schockaert, S. (2018, August). SeVeN: Augmenting Word Embeddings with Unsupervised Relation Vectors. In Proceedings of the 27th International Conference on Computational Linguistics (pp. 2653-2665)._