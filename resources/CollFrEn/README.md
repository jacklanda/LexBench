# CollFrEn: Rich Bilingual English–French Collocation Resource

This is the repository for the MWE-LEX 2020 (Coling 2020) paper: 

_Fisas, B., Espinosa-Anke, L., Codina-Filbà, J. & Wanner, L. (2020). CollFrEn: Rich Bilingual English–French Collocation Resource. In Proceedings of the Joint Workshop on Multiword Expressions and Electronic Lexicons (MWE-LEX 2020). Coling 2020._

This is a collection of English and French collocations categorized according to their _lexical function_ [1], and enriched with manual translations and where each element of the collocationo (base and collocaet) are manually disambiguated against BabelNet [2]. We also release large-scale datases containing sentences where these collocations occur, which can be used for training MWE representations, or as a resource for corpus linguistics and lexicography. We also release _relation embeddings_, i.e., vector representations for the collocations using SeVeN [3]. In the `notebooks/` folder there are examples of how to load, explore and analyze this resource.

# Citing CollFrEn

If you use this resource in your research, please use the following `bibtex` entry:

```
@inproceedings{collfren2020,
  title={CollFrEn: Rich Bilingual English–French Collocation Resource},
  author={Fisas, Beatriz and Espinosa-Anke, Luis and Codina-Filbà and Wanner, Leo},
  booktitle={ Joint Workshop on Multiword Expressions and Electronic Lexicons (MWE-LEX 2020)},
  year={2020}
}
```

# References

- [1] Mel’cuk, I. (1996). Lexical functions: a tool for the description of lexical relations in a lexicon. Lexical functions in lexicography and natural language processing, 31, 37-102.
- [2] Navigli, R., & Ponzetto, S. P. (2012). BabelNet: The automatic construction, evaluation and application of a wide-coverage multilingual semantic network. Artificial Intelligence, 193, 217-250.
- [3] Espinosa-Anke, L., & Schockaert, S. (2018, August). SeVeN: Augmenting Word Embeddings with Unsupervised Relation Vectors. In Proceedings of the 27th International Conference on Computational Linguistics (pp. 2653-2665).
