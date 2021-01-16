from query_strategies import RandomSampling, BadgeSampling, \
        BaselineSampling, LeastConfidence, MarginSampling, \
        EntropySampling, CoreSet, ActiveLearningByLearning, \
        LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
        KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
        AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

def get_strategy(args, X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform=None):
    if args.alg == 'rand': # random sampling
        strategy = RandomSampling(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'conf': # confidence-based sampling
        strategy = LeastConfidence(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'marg': # margin-based sampling
        strategy = MarginSampling(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'badge': # batch active learning by diverse gradient embeddings
        strategy = BadgeSampling(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'coreset': # coreset sampling
        strategy = CoreSet(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'entropy': # entropy-based sampling
        strategy = EntropySampling(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'baseline': # badge but with k-DPP sampling instead of k-means++
        strategy = BaselineSampling(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)
    elif args.alg == 'albl': # active learning by learning
        albl_list = [LeastConfidence(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform),
            CoreSet(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform)]
        strategy = ActiveLearningByLearning(args,X_tr, Y_tr, idxs_lb, net, handler, tr_transform, te_transform, strategy_list=albl_list, delta=0.1)
    else:
        print('choose a valid acquisition function', flush=True)
        raise ValueError
    return strategy