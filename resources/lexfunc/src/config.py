params = {

	'diffvec_file':'data/diffvec/diffvec.csv',
	'collocations_folder':'data/collocations',
	'word_vectors':'',
	'relation_vectors':'',

	'min_label_freq':99,

	'composition':'leftw', # valid: diff, sum, concat, mult, leftw

	# valid choices = ['diffvec'], ['collocations'] or ['diffvec', 'collocations']
	'data_choice':['collocations'],

	'USE_SEVEN':True,

	'experiment':'split' # 'split' for 2/3 train or 'cv' for k-fold cross validation

}