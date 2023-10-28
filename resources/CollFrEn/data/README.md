# CollFrEn Collocations

This README text describes the information contained in the ENGLISH (Section 1) and FRENCH (Section 2) dataset of disambiguated syntagmatic lexical functions.

## ENGLISH DISAMBIGUATED SYNTAGMATIC LEXICAL FUNCTIONS DATASET

This dataset is a collection of 7,480 syntagmatic lexical function instances in English, whose keyword has been disambiguated with its Babelnetid.
From the original I.Melcuk's dataset, only the most productive lexical functions have been included here. All the selected lexical functions have more than 30 instances. 

All the information in BLACK refers to the Keyword, while the information in RED refers to the Value.

The dataset includes:
1. The lexical function

2. The keyword

3. The keyword's part of speech

4. Keyword_comments
   The original Melcuk's comments about the keyword, which have been classified into diferent categories: SEM (semantic information), SX (syntactic information), USE (if the comment refers to its literal or metaphoric use) and EX (if it is an example).
  
  EXAMPLES:
     SEM:[pronunc.]    AntiMagn(accent) = imperceptible
     SX:[with N-II]    QSyn(agree) = _go along_
     USE:[metaphoric]  AntiBon(atmosphere) = evil

5. Value_comments
   The original Melcuk's comments about the value, also classified into:  REG (if it refers to the formal/informal, old-fashioned, childish, literary or slang register), VAR (information about an english or american use), and POSIT (when the comment informs about the anteposed or postposed position of the value with respect to the keyword whenever this might not be the non-marked position). Other non-classified comments are restrictions imposed to the lexical function.

  EXAMPLES:
    - REG:old-fashioned   AntiMagn(crazy) = dotty
    - VAR:British         Qsyn(commotion) = kerfuffle
    - POSIT:anteposed     Magn(appreciate) = very much, meaning that this lexical function would be instantiated as "very much appreciate" where the Value (very much) is anteposed to the Keyword (appreciate).
    - POSIT=postposed     (Keyword=battle Value=royal) would indicate that the LF would be "battle royal" were the value is postposed to the keyword
    - Other restrictions: Son(bird) = chirps In this case, the Value_comments column includes the restriction "B.is small" indicating that this value is only valid if the bird producing the sound is small.

6. The value of the Lexical Function

7. Subcategorisation pattern
   This information has been split in 2 columns, one to the left of the column "Value" (if the arguments are anteposed to the value) and one to the right of the column "Value" (if the arguments are postposed to the value or intraposed in the value). In the vast majority of the cases, the arguments are postposed. If the arguments are inserted in the value, they are substituted by "[...]" in the column Value and the whole expression (value + arguments) is explicited in the Subcategorisation column.

  EXAMPLES:
    - anteposed subcategorisation:   Magn(try) = [N-I(gen)] best
    - postposed subcategorisation:   CausOper1(available) = make [N-I ~]
    - intraposed subcategorisation:  Anti(answer) = _give [N-II] the runaround_  

    where N-I is the first argument, N-II is the second argument and ~ is the keyword.

8. Fusion (//)
  If the value of a lexical function expresses by itself both the sense of the keyword and the lexical function it is marked with //. Although the Lexical function is syntagmatic in its nature, in these cases it behaves as a paradigmatic one.
  
  EXAMPLE:
    - Magn(agree) = _there is no daylight_ between NX and NY  // in comparison to the normal operation of Magn
    - Magn(agree) = completely

9. Intensity degree 
  The operator < indicates the degree of synonymy.  
  
  EXAMPLE
     - Magn(amazing) = jaw-dropping  <<
     - Magn(amazing) = stunning < 
     - In this example, these operators << and < mean that "jaw-dropping" is even more intense than "stunning".

10. Columns Babelnetid and Babelnetid2 have the Babelnet identifiers that correspond to the sense of the keyword in each particular instance of the lexical function. The first one (Babelnetid) is the main or prefered sense, and a proposal for a second or alternative id (Babelnetid2) is added whenever the disambiguation in between the sense choices in Babelnet is not possible without context, when two synsets are practically identical, or when two senses are possible (for example, a literal sense and a metaphorical sense).
  
   EXAMPLES:
   - A0(biology) = biological 	keyw_babelnetId=bn:00010549n keyw_babelnetId_2=bn:00010543n
   - CausFact0(bomb) = explode   keyw_babelnetId=bn:00011917n keyw_babelnetId_2=bn:01237332n 
   This keyword has a semantic comment indicating that it refers to airplanes. The first babelnetId is a more general sense and the second Id is specifically an aerial bomb, so both are possible. 
   - Magn(thirst) = unquenchable keyw_babelnetId=bn:00076968n keyw_babelnetId_2=bn:00045230n. 
      The first babelnetId refers to a physical need (A physiological need to drink) while the second Id a more is a metaphorical use of the keyword (Strong desire for something (not food or drink)). The lexical function Magn could be applied to both senses with the same value.

11. Comments to the Babelnetids.

    Extra-information is recorded
    
    11a. if some multiwords have their own proper Babelnetid (in red if they are collocations, and in black if the keyword is a multiword or contains one)
       EXAMPLE:
       - Magn(thin) = _as a rake_  
       - Babelnet has a specific id for the whole expression and it is noted in this column for comments (Bid for "thin as a rake" bn:13662369a)
    11b. if there is a mismatch between the keyword's part of speech and the synset's part of speech (Most cases are past participles).
       EXAMPLE:
       - Magn(staked (pos=A)) = heavily    
       - Although the Keyword_pos is A (for adjective), Babelnet only includes an id for the corresponding verb (to stake). Staked is in fact the past participle of "to stake", so this verbal id is noted in the keyw_babelnetid column (bn:00082409v) and a comment "past participle" is written in the Comments Bid column.
    11c. in a few cases, a comment is included if the keyword (for example: "bift") or the specific sense in which the keyword is used (for example: "conscience" in the sense of perception) doesn't have a Babelnetid.

**Abbreviations used in the dataset**:
- pl = plural
- gen = genitive
- Vger = gerund
- Vinf = infinitive
- Aposs = possessive adjective
- _multiword phrase_

## FRENCH DISAMBIGUATED SYNTAGMATIC LEXICAL FUNCTIONS DATASET

This dataset is a collection of 6,733 syntagmatic lexical function instances in French whose keyword has been disambiguated with its Babelnetid. For every instance keyword-value in French, its equivalent keyword-value in English has also been included.

From the original I.Melcuk's dataset, only the most productive lexical functions have been included here. All the selected lexical functions have more than 30 instances. 
All the information in BLACK refers to the Keyword, while the information in RED refers to the Value.

The dataset includes:
1. The lexical function

2. The keyword

3. The translated keyword in English

4. The keyword's part of speech

5. Keyword_comments
   The original Melcuk's comments about the keyword, which have been classified into diferent categories: SEM (semantic information), SX (syntactic information), MORPH (morphological information), USE (if the comment refers to its literal or metaphoric use), MW (if the keyword is a multiword expression) and EX (if it is an example).
  
  EXAMPLES:
     
     SEM:[prononciation]  	FinOper(accent) = perdre
     SX:[de N]    	  	Magn(beaucoup) = _à gogo_
     MORPH:[pl]           	Magn(amis) = inséparables
     USE:[métaph.]        	Bon(chapitre) = glorieux
     MW			  	Oper1(_en fleur_) = être
     

6. Value_comments
   The original Melcuk's comments about the value, also classified into:  REG (if it refers to the formal/informal, old-fashioned, childish, familiar, literary or slang register), VAR (information about an belgian or suisse use), and POSIT (when the comment informs about the anteposed or postposed position of the value with respect to the keyword whenever this might not be the non-marked position). As in the Keyword comments, SEM and SX indicate some semantic or syntactic information, MW stands for multiword and EX tags an example. Other non-classified comments are restrictions imposed to the lexical function.

  EXAMPLES:

    REG:argot		   	AntiBon(viande) = barbaque
    VAR:Belgique, Suisse   	CausFunc1(feu)  = bouter
    POSIT:anteposé 	   	Magn(admiration)= la plus grande, meaning that this lexical function would be instantiated as "la plus grande admiration" where the Value (la plus grande) is anteposed to the Keyword (admiration).
    POSIT:postposé	  	Magn(absence) = totale, meaning that this lexical function would be instantiated as "absence totale" where the Value (totale) is postposed to the Keyword (absence).
    SEM:[un homme]         	Magn(beau) = comme Apollon
    SX:[C. a un modificateur]  Oper(crainte) = éprouver
    MW				Magn(défaite) = _sur toute la ligne_
    EX:[... au deuil qui frappe sa famille ...]  Func1(deuil) = frappe
    
    Other restrictions: Magn(cri) = d’orfraie In this case, the Value_comments column includes the restriction "seulement avec pousser [*émettre]" indicating that this value is only valid with the verb POUSSER, in the sense of "emit".

7. The value of the Lexical Function

8. The translated value in English
   Two colummns are allocated for the translated value, as in many cases more than one english word is possible.

9. Subcategorisation pattern
   This information has been split in 2 columns ("anteposed subcategorisation" and "postposed & intraposed subcategorisation"), one to the left of the column "Value" (if the arguments are anteposed to the value) and one to the right of the column "Value" (if the arguments are postposed to the value or intraposed in the value). In the vast majority of the cases, the arguments are postposed. If the arguments are inserted in the value, they are substituted by "[...]" in the column Value and the whole expression (value + arguments) is explicited in the Subcategorisation column.

  EXAMPLES:
    anteposed subcategorisation:   Magn(beacoup) = [Ns] _à gogo_
    postposed subcategorisation:   Oper2(invitation) = recevoir [ART ~ de NX]
    intraposed subcategorisation:  Magn(pleurer) = toutes les larmes de [Aposs-X] corps

    where NX is the first argument and ~ is the keyword.

10. Fusion (//)
    If the value of a lexical function expresses by itself both the sense of the keyword and the lexical function it is marked with //. Although the Lexical function is syntagmatic in its nature, in these cases it behaves as a paradigmatic one.
  
  EXAMPLE:
    Magn(cher) = _hors de prix_   //
    in comparison to the normal operation of Magn
    Magn(chauve) = comme un oeuf

  
11. Intensity degree 
    The operator < indicates the degree of synonymy.  
  
  EXAMPLE
    Magn(difficulté) = énorme  	grande <
    Magn(difficulté) = sans pareil  	grande <<
    In this example, these operators << and < mean that "sans pareil" is even more intense than "énorme", and both are more intense than "grande"

12. Columns Babelnetid and Babelnetid2 have the Babelnet identifiers that correspond to the sense of the keyword in each particular instance of the lexical function. The first one (Babelnetid) is the main or prefered sense, and a proposal for a second or alternative id (Babelnetid2) is added whenever the disambiguation in between the sense choices in Babelnet is not possible without context, when two synsets are practically identical, or when two senses are possible (for example, a literal sense and a metaphorical sense).
  
  EXAMPLES:
    Magn(accepter) = de plein gré	keyw_babelnetId=bn:00082476v	keyw_babelnetId_2=bn:00082199v
    Both senses are very similar: bn:00082476v "Consent or assent to a condition, or agree to do something" and bn:00082199v "Give an affirmative reply to; respond favorably to"

    Oper1(agression) = commettre 	bn:00002015n	bn:00002016n
    The first babelnetId is a very general sense  "Le terme agression dérive du latin adgredi signifiant « aller vers », « attaquer », « marcher de l'avant. Comportement, d''attaque ou     d'opposition, non provoqué s'exprimant avec violence et brutalité», while the second Id is much more specific "Violent action that is hostile and usually unprovoked" but both can be assigned to the keyword.

    Real1(bombe) = lancer   keyw_babelnetId=bn:00011917n keyw_babelnetId_2=bn:01237332n 
    This keyword has a semantic comment indicating that it refers to airplanes. The first babelnetId is a more general sense and the second Id is specifically an aerial bomb, so both are possible. 
    
    Oper1(soif) = avoir  keyw_babelnetId=bn:00076968n	 keyw_babelnetId_2=bn:00045230n 
    The first babelnetId refers to a physical need (A physiological need to drink) while the second Id a more is a metaphorical use of the keyword (Strong desire for something (not food or drink)). The lexical function Oper1 could be applied to both senses with the same value.

13. Comments to the Babelnetids.
    Extra-information is recorded:

    13a. if there is a mismatch between the keyword's part of speech and the synset's part of speech (Most cases are past participles).
       EXAMPLE:
       Magn(influencé (pos=A)) = fortement   
       Although the Keyword_pos is A (for adjective), Babelnet does not include a synset for "influencé" as an adjective but does have a synset for the corresponding verb (influencer). "Influencé" is in fact the past participle of "influencer", so this verbal id is noted in the keyw_babelnetid column (bn:00082304v) and a comment "past participle" is written in the Comments Bid column.
    13b. if the keyword is not recorded in Babelnet or the specific sense in which the keyword is used doesn't have a BabelnetId, this columns includes a comment "not in Bnet" or "this sense is not in Bnet".
    13c. in some cases, more than 2 BabelnetIds are possible, and comments such as "also possible",  "other Bids are possible" are added in this column.
    13d. Finally if there is not enough context to disambiguate the keyword's synset in Babelnet, a remark such as "needs more context" or "a more general sense is needed in Bnet" "Bnet does not have the fig sense".

**Abbreviations used in the dataset**:
- pl = plural
- gen = genitive
- Vger = gerund
- Vinf = infinitive
- Aposs = possessive adjective
- _multiword phrase_
