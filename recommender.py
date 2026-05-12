import math
from DBManager import get_all_cannabinoids, get_all_terpenes, get_user_sessions

class KNN:
    def __init__(self, user_id):
        self.user_id = user_id
        
        cannabinoids = get_all_cannabinoids()
        terpenes = get_all_terpenes()
        
        self.cannabinoidMap = {}
        for i, c in enumerate(cannabinoids):
            self.cannabinoidMap[c['id']] = i
        
        self.terpeneMap = {}
        for i, t in enumerate(terpenes):
            self.terpeneMap[t['id']] = i
        
        #Starting weights
        self.w_chem = 0.5   # w0
        self.w_dose = 0.3   # w1
        self.w_admin = 0.2  # w2
        self.gamma = 0.1    # tolerance
        
        self.history = get_user_sessions(user_id)
        self._tuning_logic()

    def _tuning_logic(self):
        count = len(self.history)
        if count < 10:
            return

        #parameters for grid search
        w_chem_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
        w_dose_vals = [0.1, 0.2, 0.3, 0.4]
        gamma_vals = [0.01, 0.05, 0.1, 0.2, 0.5]

        best_score = -1.0
        best_params = (self.w_chem, self.w_dose, self.w_admin, self.gamma)
        
        original_history = self.history
        
        #Choose prediction strategy
        is_mature = False
        if count >= 20:
            is_mature = True
        
        for wc in w_chem_vals:
            for wd in w_dose_vals:
                if wc + wd >= 1.0:
                    continue

                wa = 1.0 - wc - wd
                for g in gamma_vals:
                    total_f1 = 0.0
                    iterations = 0
                    
                    #Temporary set weights
                    self.w_chem, self.w_dose, self.w_admin, self.gamma = wc, wd, wa, g
                    
                    if not is_mature:
                        #Leave-One-Out Cross-Validation
                        for i in range(count):
                            test_session = original_history[i]
                            self.history = original_history[:i] + original_history[i+1:]
                            
                            f1 = self._evaluate_session(test_session)
                            total_f1 += f1
                            iterations += 1
                    else:
                        for i in range(10, count):
                            test_session = original_history[i]
                            self.history = original_history[:i]
                            
                            f1 = self._evaluate_session(test_session)
                            total_f1 += f1
                            iterations += 1
                    
                    if iterations > 0:
                        avg_f1 = total_f1 / iterations
                    else:
                        avg_f1 = 0

                    if avg_f1 > best_score:
                        best_score = avg_f1
                        best_params = (wc, wd, wa, g)
        
        #Apply best found parameters
        self.w_chem, self.w_dose, self.w_admin, self.gamma = best_params
        self.history = original_history

    def _evaluate_session(self, session):
        query_prod = {
            'cannabinoids': session['cannabinoids'],
            'terpenes': session['terpenes'],
            'type': session.get('Products', {}).get('type')
        }
        query_dose = session.get('servings', 0)
        
        preds = self.predict(query_prod, query_dose, return_all=True)
        return self._calculate_f1(session.get('effects', []), preds)

    def _calculate_f1(self, actual_effects, predicted_results):
        #Calculates F1 score for multi-label prediction with 0.5 threshold.
        predictedTags = set()
        for res in predicted_results:
            if res['probability'] >= 0.5:
                predictedTags.add(res['effect'])
                
        actual_tags = set(actual_effects)
        
        if not predictedTags:
            return 0.0 if actual_tags else 1.0
        
        if not actual_tags:
            return 0.0 #Predicted something when nothing was there
            
        tp = len(actual_tags.intersection(predictedTags))
        precision = tp / len(predictedTags) 
        recall = tp / len(actual_tags)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * ((precision * recall) / (precision + recall))

    def _vectorize(self, data):
        #Cannabinoids (Continuous values, normalized)
        c_vec = [0.0] * len(self.cannabinoidMap)
        for pc in data.get('cannabinoids', []):
            cid = pc.get('cannabinoid_id')

            if cid in self.cannabinoidMap:
                c_vec[self.cannabinoidMap[cid]] = pc.get('serving', 0.0)
        
        #Normalize
        mag = math.sqrt(sum(x*x for x in c_vec))
        if mag > 0:
            c_vec = [x/mag for x in c_vec]

        #Terpenes (Multi-hot binary encoding)
        t_vec = [0.0] * len(self.terpeneMap)
        for pt in data.get('terpenes', []):
            tid = pt.get('Terpene_id') or pt.get('terpene_id')

            if tid in self.terpeneMap:
                t_vec[self.terpeneMap[tid]] = 1.0
                
        return c_vec + t_vec

    def _calculate_similarity(self, query_prod, query_dose, session):
        #Chemical Similarity (Cosine)
        v1 = self._vectorize(query_prod)
        v2 = self._vectorize(session)
        
        dot = 0
        for a, b in zip(v1, v2):
            dot += a * b

        m1 = 0
        for a in v1:
            m1 += a*a
        m1 = math.sqrt(m1)

        m2 = 0
        for b in v2:
            m2 += b*b
        m2 = math.sqrt(m2)

        if(m1 * m2) > 0:
            s_chem = dot / (m1 * m2)
        else:
            s_chem = 0.0

        #Dose Similarity (Gaussian RBF)
        d1 = query_dose
        d2 = session.get('servings', 0)
        s_dose = math.exp(-self.gamma * (d1 - d2)**2)

        #Admin Route Similarity (Exact Match)
        a1 = query_prod.get('type')
        a2 = session.get('Products', {}).get('type')
        
        if a1 == a2:
            s_admin = 1.0
        else:            
            s_admin = 0.0

        return (self.w_chem * s_chem) + (self.w_dose * s_dose) + (self.w_admin * s_admin)

    def predict(self, query_product, query_dose, k=5, return_all=False):
        if not self.history:
            return []

        #Fetch & Rank Neighbors
        neighbors = []
        for session in self.history:
            sim = self._calculate_similarity(query_product, query_dose, session)
            neighbors.append((sim, session.get('effects', [])))
        
        neighbors.sort(key=lambda x: x[0], reverse=True)
        #Filter weak matches
        topK = []
        for n in neighbors[:k]:
            if n[0] > 0.1:
                topK.append(n)

        if not topK:
            return []

        #Aggregate Labels
        effect_scores = {}

        totalSim = 0
        for n in topK:
            totalSim += n[0]

        for sim, effects in topK:
            for effect in effects:
                effect_scores[effect] = effect_scores.get(effect, 0) + sim

        #Format Results
        results = []
        for e, s in effect_scores.items():
            results.append({'effect': e, 'probability': s / totalSim})

        results.sort(key=lambda x: x['probability'], reverse=True)
        
        if return_all:
            return results
        
        return results[:3]