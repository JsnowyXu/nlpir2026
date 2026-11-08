"""Shared causal org-candidate features for classical forecast baselines."""
import numpy as np
from scipy import sparse


FEATURE_NAMES = ['log_pair_count','log_recent_count','log_previous_count','log_decayed_count',
                 'personal_share','recent_personal_share','recent_change','years_since_pair',
                 'global_share','global_recent_share','global_change','log_org_events',
                 'org_distinct_labels','org_age_years','class_personal_share','section_personal_share']


class PairFeatures:
    def __init__(self, dataset, cutoff):
        self.data = dataset
        self.cutoff = cutoff
        self.history = dataset.history(cutoff)
        self.recent = dataset.counts(cutoff)
        self.previous = dataset.counts(cutoff-1)
        self.decayed = dataset.history(cutoff,half_life=2.)
        self.total = np.asarray(self.history.sum(axis=1)).ravel()
        self.recent_total = np.asarray(self.recent.sum(axis=1)).ravel()
        self.global_counts = np.asarray(self.history.sum(axis=0)).ravel()
        self.global_recent = np.asarray(self.recent.sum(axis=0)).ravel()
        self.global_previous = np.asarray(self.previous.sum(axis=0)).ravel()
        self.distinct = np.diff(self.history.indptr)
        self.lag = sparse.csr_matrix(self.history.shape,dtype=np.float32)
        # Store last year relative to a nonzero origin, no future target-year data.
        self.origin = min(dataset.manifest['years'])-1
        for year in dataset.manifest['years']:
            if year<=cutoff:
                current=dataset.counts(year).sign()*(year-self.origin)
                self.lag=self.lag.maximum(current)
        self.parents = {}
        for width in [1,3]:
            names=sorted({c[:width] for c in dataset.codes})
            ids={c:i for i,c in enumerate(names)}
            mapping=np.array([ids[c[:width]] for c in dataset.codes])
            projection=sparse.coo_matrix((np.ones(len(mapping)),(np.arange(len(mapping)),mapping)),
                                          shape=(len(mapping),len(names))).tocsr()
            self.parents[width]=(self.history@projection,mapping)

    def pairs(self,orgs,labels):
        orgs,labels=np.asarray(orgs),np.asarray(labels)
        def pair(matrix):
            return np.asarray(matrix[orgs,labels]).ravel()
        hist,recent,previous,decay=[pair(m) for m in [self.history,self.recent,self.previous,self.decayed]]
        total=np.maximum(self.total[orgs],1.)
        rtotal=np.maximum(self.recent_total[orgs],1.)
        last=pair(self.lag)
        lag=np.where(last>0,self.cutoff-self.origin-last,30.)
        values=[np.log1p(hist),np.log1p(recent),np.log1p(previous),np.log1p(decay),
                hist/total,recent/rtotal,np.log1p(recent)-np.log1p(previous),lag,
                self.global_counts[labels]/max(self.global_counts.sum(),1.),
                self.global_recent[labels]/max(self.global_recent.sum(),1.),
                np.log1p(self.global_recent[labels])-np.log1p(self.global_previous[labels]),
                np.log1p(total),self.distinct[orgs],self.cutoff-self.data.first_year[orgs]+1]
        for width in [3,1]:
            counts,mapping=self.parents[width]
            values.append(np.asarray(counts[orgs,mapping[labels]]).ravel()/total)
        return np.stack(values,axis=1).astype(np.float32)

    def score_model(self,model,orgs):
        c=self.data.candidate_ids
        features=self.pairs(np.repeat(orgs,len(c)),np.tile(c,len(orgs)))
        if hasattr(model,'predict_proba'):
            values=model.predict_proba(features)[:,1]
        else:
            values=model.decision_function(features)
        return values.reshape(len(orgs),len(c))


def sampled_training_pairs(data,seed,max_pairs=300000,negatives_per_query=10):
    rng=np.random.RandomState(seed)
    all_features,all_labels=[],[]
    years=data.split_years('train')
    if not years:
        raise ValueError('No supervised training transitions')
    per_year=max_pairs//len(years)
    for year in years:
        orgs,truth=data.queries(year)
        candidates=data.candidate_ids
        features=PairFeatures(data,year-1)
        sources,destinations,labels=[],[],[]
        # Query sampling independent of future activity; positives are only labels.
        query_budget=max(1,per_year//(negatives_per_query+2))
        chosen=np.sort(rng.choice(len(orgs),min(query_budget,len(orgs)),replace=False))
        for i in chosen:
            positives=np.intersect1d(candidates,truth[i])
            negative_pool=np.setdiff1d(candidates,positives,assume_unique=True)
            negatives=rng.choice(negative_pool,min(negatives_per_query,len(negative_pool)),replace=False)
            target=np.r_[positives,negatives]
            sources.extend([orgs[i]]*len(target));destinations.extend(target)
            labels.extend([1]*len(positives)+[0]*len(negatives))
        x=features.pairs(sources,destinations)
        y=np.asarray(labels,dtype=np.int8)
        if len(y)>per_year:
            select=rng.choice(len(y),per_year,replace=False)
            x,y=x[select],y[select]
        all_features.append(x);all_labels.append(y)
    return np.concatenate(all_features),np.concatenate(all_labels)
