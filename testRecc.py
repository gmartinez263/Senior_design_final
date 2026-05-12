import unittest
from unittest.mock import patch, MagicMock
from recommender import KNN

class TestRecommendationPhases(unittest.TestCase):
    def setUp(self):
        #Mock data for Cannabinoids and Terpenes
        self.mock_cannabinoids = [
            {'id': 1, 'name': 'THC'},
            {'id': 2, 'name': 'CBD'},
            {'id': 3, 'name': 'CBG'},
            {'id': 4, 'name': 'CBN'},
            {'id': 5, 'name': 'THCV'},
            {'id': 6, 'name': 'CBDV'},
            {'id': 7, 'name': 'CBC'}
        ]
        self.mock_terpenes = [
            {'id': 1, 'name': 'Myrcene'},
            {'id': 2, 'name': 'Limonene'},
            {'id': 3, 'name': 'Pinene'},
            {'id': 4, 'name': 'Linalool'},
            {'id': 5, 'name': 'Caryophyllene'},
            {'id': 6, 'name': 'Humulene'},
            {'id': 7, 'name': 'Terpinolene'},
            {'id': 8, 'name': 'Ocimene'}
        ]
        
        #Helper to create a dummy session
        self.create_session = lambda sid, product_type, servings, effects, c_list=None, t_list=None: {
            'id': sid,
            'product_id': sid, 
            'servings': servings,
            'Products': {'type': product_type},
            'cannabinoids': c_list if c_list is not None else [{'cannabinoid_id': 1, 'serving': 10.0}],
            'terpenes': t_list if t_list is not None else [{'terpene_id': 1}],
            'effects': effects
        }

    def print_step(self, step_name, description):
        print(f"\n[TEST STEP] {step_name}")
        print(f"Description: {description}")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_phase_1_cold_start_prediction(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Cold Start (< 10 sessions) with diverse history
        self.print_step("Phase 1: Cold Start", "Testing prediction with 6 distinct sessions and various tags.")
        
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        #6 sessions with different chemical profiles
        mock_sessions.return_value = [
            self.create_session(1, 'Edible', 1.0, ['Relaxed', 'Happy'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 1}]), # THC + Myrcene
            self.create_session(2, 'Edible', 1.2, ['Relaxed', 'Hungry'], [{'cannabinoid_id': 1, 'serving': 15.0}, {'cannabinoid_id': 3, 'serving': 2.0}], [{'terpene_id': 1}, {'terpene_id': 2}]), # THC/CBG + Myrcene/Limonene
            self.create_session(3, 'Tincture', 0.5, ['Focused'], [{'cannabinoid_id': 2, 'serving': 20.0}], [{'terpene_id': 3}]), # CBD + Pinene
            self.create_session(4, 'Tincture', 0.6, ['Focused', 'Calm'], [{'cannabinoid_id': 2, 'serving': 25.0}, {'cannabinoid_id': 6, 'serving': 5.0}], [{'terpene_id': 3}, {'terpene_id': 4}]), # CBD/CBDV + Pinene/Linalool
            self.create_session(5, 'Capsule', 1.0, ['Sleepy'], [{'cannabinoid_id': 4, 'serving': 5.0}], [{'terpene_id': 1}, {'terpene_id': 5}]), # CBN + Myrcene/Caryophyllene
            self.create_session(6, 'Edible', 1.0, ['Euphoric'], [{'cannabinoid_id': 1, 'serving': 20.0}, {'cannabinoid_id': 5, 'serving': 2.0}], [{'terpene_id': 2}, {'terpene_id': 7}]), # THC/THCV + Limonene/Terpinolene
        ]
        
        knn = KNN(user_id=1)
        
        #High THC Edible with Myrcene
        new_product = {
            'cannabinoids': [{'cannabinoid_id': 1, 'serving': 12.0}],
            'terpenes': [{'terpene_id': 1}],
            'type': 'Edible'
        }
        query_dose = 1.0
        
        print(f"Predicting for product type: {new_product['type']}, dose: {query_dose}")
        results = knn.predict(new_product, query_dose)
        
        predicted_effects = []
        for r in results:
            predicted_effects.append(r['effect'])
        print(f"Predicted Effects: {predicted_effects}")
        
        self.assertIn('Relaxed', predicted_effects)
        self.assertIn('Happy', predicted_effects)
        print("Success: Phase 1 correctly identified top tags from distinct history.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_phase_2_initial_tuning_prediction(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Initial Tuning (10-19 sessions) with diverse history
        self.print_step("Phase 2: Initial Tuning", "Testing prediction with 15 distinct sessions of varying products and tags.")
        
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        history = [
            self.create_session(1, 'Tincture', 0.5, ['Focused'], [{'cannabinoid_id': 2, 'serving': 10.0}], [{'terpene_id': 3}]),
            self.create_session(2, 'Tincture', 0.5, ['Focused', 'Creative'], [{'cannabinoid_id': 2, 'serving': 12.0}, {'cannabinoid_id': 3, 'serving': 1.0}], [{'terpene_id': 3}, {'terpene_id': 8}]),
            self.create_session(3, 'Tincture', 0.6, ['Focused'], [{'cannabinoid_id': 2, 'serving': 15.0}], [{'terpene_id': 3}]),
            self.create_session(4, 'Tincture', 0.4, ['Alert'], [{'cannabinoid_id': 2, 'serving': 8.0}, {'cannabinoid_id': 5, 'serving': 1.0}], [{'terpene_id': 3}, {'terpene_id': 7}]),
            self.create_session(5, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 1}]),
            self.create_session(6, 'Edible', 1.5, ['Relaxed', 'Sleepy'], [{'cannabinoid_id': 1, 'serving': 20.0}, {'cannabinoid_id': 4, 'serving': 2.0}], [{'terpene_id': 1}, {'terpene_id': 4}]),
            self.create_session(7, 'Edible', 0.8, ['Relaxed', 'Happy'], [{'cannabinoid_id': 1, 'serving': 5.0}], [{'terpene_id': 2}]),
            self.create_session(8, 'Capsule', 1.0, ['Pain Relief'], [{'cannabinoid_id': 2, 'serving': 25.0}, {'cannabinoid_id': 7, 'serving': 5.0}], [{'terpene_id': 5}]),
            self.create_session(9, 'Capsule', 1.0, ['Sleepy'], [{'cannabinoid_id': 4, 'serving': 10.0}], [{'terpene_id': 1}]),
            self.create_session(10, 'Tincture', 0.5, ['Creative'], [{'cannabinoid_id': 1, 'serving': 2.0}, {'cannabinoid_id': 2, 'serving': 10.0}], [{'terpene_id': 8}]),
            self.create_session(11, 'Edible', 1.0, ['Hungry'], [{'cannabinoid_id': 1, 'serving': 15.0}], [{'terpene_id': 6}]),
            self.create_session(12, 'Tincture', 1.0, ['Calm'], [{'cannabinoid_id': 2, 'serving': 50.0}], [{'terpene_id': 4}]),
            self.create_session(13, 'Capsule', 2.0, ['Heavy Sleep'], [{'cannabinoid_id': 1, 'serving': 10.0}, {'cannabinoid_id': 4, 'serving': 10.0}], [{'terpene_id': 1}, {'terpene_id': 6}]),
            self.create_session(14, 'Tincture', 0.2, ['Microdose Happy'], [{'cannabinoid_id': 1, 'serving': 2.5}], [{'terpene_id': 2}]),
            self.create_session(15, 'Edible', 1.0, ['Relaxed', 'Calm'], [{'cannabinoid_id': 1, 'serving': 5.0}, {'cannabinoid_id': 2, 'serving': 5.0}], [{'terpene_id': 4}])
        ]
            
        mock_sessions.return_value = history
        
        knn = KNN(user_id=1)
        
        #New Tincture focused on CBD + Pinene
        new_product = {
            'cannabinoids': [{'cannabinoid_id': 2, 'serving': 11.0}],
            'terpenes': [{'terpene_id': 3}],
            'type': 'Tincture'
        }
        query_dose = 0.5
        
        print(f"Predicting for product type: {new_product['type']}, dose: {query_dose}")
        results = knn.predict(new_product, query_dose)
        
        predicted_effects = []
        for r in results:
            predicted_effects.append(r['effect'])
        print(f"Predicted Effects: {predicted_effects}")
        
        self.assertIn('Focused', predicted_effects)
        print("Success: Phase 2 predicted correctly with tuned parameters on distinct data.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_phase_3_mature_prediction(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Mature Phase (>= 20 sessions) with diverse history
        self.print_step("Phase 3: Mature Phase", "Testing prediction with 21 distinct sessions.")
        
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        history = [
            self.create_session(1, 'Capsule', 1.0, ['Sleepy'], [{'cannabinoid_id': 4, 'serving': 5.0}], [{'terpene_id': 1}]),
            self.create_session(2, 'Capsule', 1.0, ['Pain Relief'], [{'cannabinoid_id': 2, 'serving': 20.0}], [{'terpene_id': 5}]),
            self.create_session(3, 'Capsule', 1.1, ['Sleepy', 'Pain Relief'], [{'cannabinoid_id': 2, 'serving': 10.0}, {'cannabinoid_id': 4, 'serving': 5.0}], [{'terpene_id': 1}, {'terpene_id': 5}]),
            self.create_session(4, 'Edible', 2.0, ['Euphoric'], [{'cannabinoid_id': 1, 'serving': 20.0}], [{'terpene_id': 2}]),
            self.create_session(5, 'Edible', 2.2, ['Euphoric', 'Giggles'], [{'cannabinoid_id': 1, 'serving': 25.0}], [{'terpene_id': 2}, {'terpene_id': 7}]),
            self.create_session(6, 'Tincture', 0.25, ['Energetic'], [{'cannabinoid_id': 1, 'serving': 2.5}, {'cannabinoid_id': 5, 'serving': 1.0}], [{'terpene_id': 7}]),
            self.create_session(7, 'Tincture', 0.3, ['Focused'], [{'cannabinoid_id': 2, 'serving': 5.0}, {'cannabinoid_id': 3, 'serving': 2.0}], [{'terpene_id': 3}]),
            self.create_session(8, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}, {'cannabinoid_id': 2, 'serving': 10.0}], [{'terpene_id': 1}]),
            self.create_session(9, 'Capsule', 1.0, ['Anti-inflammatory'], [{'cannabinoid_id': 7, 'serving': 10.0}], [{'terpene_id': 6}]),
            self.create_session(10, 'Tincture', 0.5, ['Creative'], [{'cannabinoid_id': 1, 'serving': 5.0}, {'cannabinoid_id': 3, 'serving': 5.0}], [{'terpene_id': 8}]),
            self.create_session(11, 'Edible', 1.5, ['Heavy'], [{'cannabinoid_id': 1, 'serving': 30.0}], [{'terpene_id': 1}]),
            self.create_session(12, 'Tincture', 1.0, ['Calm'], [{'cannabinoid_id': 2, 'serving': 25.0}, {'cannabinoid_id': 6, 'serving': 10.0}], [{'terpene_id': 4}]),
            self.create_session(13, 'Capsule', 0.5, ['Mild Relief'], [{'cannabinoid_id': 2, 'serving': 10.0}], [{'terpene_id': 5}]),
            self.create_session(14, 'Edible', 1.0, ['Happy'], [{'cannabinoid_id': 1, 'serving': 10.0}, {'cannabinoid_id': 5, 'serving': 1.0}], [{'terpene_id': 2}]),
            self.create_session(15, 'Tincture', 0.75, ['Productive'], [{'cannabinoid_id': 2, 'serving': 15.0}, {'cannabinoid_id': 5, 'serving': 2.0}], [{'terpene_id': 3}, {'terpene_id': 7}]),
            self.create_session(16, 'Edible', 0.5, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 5.0}], [{'terpene_id': 1}]),
            self.create_session(17, 'Capsule', 1.0, ['Restorative'], [{'cannabinoid_id': 2, 'serving': 20.0}, {'cannabinoid_id': 7, 'serving': 5.0}], [{'terpene_id': 6}]),
            self.create_session(18, 'Tincture', 0.5, ['Grounded'], [{'cannabinoid_id': 3, 'serving': 10.0}], [{'terpene_id': 8}]),
            self.create_session(19, 'Edible', 2.0, ['Giggly'], [{'cannabinoid_id': 1, 'serving': 20.0}, {'cannabinoid_id': 7, 'serving': 2.0}], [{'terpene_id': 7}]),
            self.create_session(20, 'Capsule', 1.5, ['Deep Sleep'], [{'cannabinoid_id': 4, 'serving': 15.0}], [{'terpene_id': 1}, {'terpene_id': 4}]),
            self.create_session(21, 'Tincture', 0.25, ['Clear-headed'], [{'cannabinoid_id': 2, 'serving': 10.0}, {'cannabinoid_id': 5, 'serving': 0.5}], [{'terpene_id': 3}])
        ]
            
        mock_sessions.return_value = history
        
        knn = KNN(user_id=1)
        
        #High CBN Capsule for sleep
        new_product = {
            'cannabinoids': [{'cannabinoid_id': 4, 'serving': 12.0}],
            'terpenes': [{'terpene_id': 1}],
            'type': 'Capsule'
        }
        query_dose = 1.0
        
        print(f"Predicting for product type: {new_product['type']}, dose: {query_dose}")
        results = knn.predict(new_product, query_dose)
        
        predicted_effects = []
        for r in results:
            predicted_effects.append(r['effect'])
        print(f"Predicted Effects: {predicted_effects}")
        
        self.assertIn('Sleepy', predicted_effects)
        print("Success: Phase 3 handled mature, distinct history correctly.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_edge_case_no_terpenes(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Edge Case: Little to no terpene data
        self.print_step("Edge Case: No Terpenes", "Testing prediction when the new product has no terpene data.")
        
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        mock_sessions.return_value = [
            self.create_session(1, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 1}]),
            self.create_session(2, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 2}]),
            self.create_session(3, 'Edible', 1.1, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 1}]),
            self.create_session(4, 'Edible', 0.9, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 2}]),
            self.create_session(5, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [{'terpene_id': 1}]),
        ]
        
        knn = KNN(user_id=1)
        
        #product has no terpenes
        new_product = {
            'cannabinoids': [{'cannabinoid_id': 1, 'serving': 10.0}],
            'terpenes': [], #empty terpenes
            'type': 'Edible'
        }
        query_dose = 1.0
        
        print(f"Predicting for product with NO terpenes, type: {new_product['type']}")
        results = knn.predict(new_product, query_dose)
        
        predicted_effects = []
        for r in results:
            predicted_effects.append(r['effect'])
        print(f"Predicted Effects: {predicted_effects}")
        
        #Should still predict based on cannabinoids and product type
        self.assertIn('Relaxed', predicted_effects)
        print("Success: Correctly predicted 'Relaxed' despite missing terpene data.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_dose_sensitivity(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Test how dose difference affects similarity
        self.print_step("Dose Sensitivity", "Testing if a large dose discrepancy reduces similarity for identical products.")
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        # History: 1.0 serving leads to 'Relaxed'
        mock_sessions.return_value = [self.create_session(1, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], [])]
        knn = KNN(user_id=1)
        
        new_product = {'cannabinoids': [{'cannabinoid_id': 1, 'serving': 10.0}], 'terpenes': [], 'type': 'Edible'}
        
        #1st case: Identical dose (1.0)
        results_exact = knn.predict(new_product, 1.0)
        #2nd case: greater dose (10.0)
        results_diff = knn.predict(new_product, 10.0)
        
        print(f"Results exact: {results_exact}, Results diff: {results_diff}")
        self.assertTrue(len(results_exact) >= len(results_diff))
        print("Success: Dose sensitivity test completed.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_high_cbd_vs_thc(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Test distinguishing between CBD and THC profiles
        self.print_step("CBD vs THC Profile", "Testing if the engine distinguishes between high-CBD and high-THC products.")
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        mock_sessions.return_value = [
            self.create_session(1, 'Edible', 1.0, ['Psychoactive'], [{'cannabinoid_id': 1, 'serving': 20.0}], []), # THC
            self.create_session(2, 'Edible', 1.0, ['Calm'], [{'cannabinoid_id': 2, 'serving': 20.0}], []) # CBD
        ]
        knn = KNN(user_id=1)
        
        #High CBD product
        new_product = {
            'cannabinoids': [{'cannabinoid_id': 2, 'serving': 20.0}],
            'terpenes': [], 'type': 'Edible'
        }
        results = knn.predict(new_product, 1.0)
        
        predicted_effects = []
        for r in results:
            predicted_effects.append(r['effect'])
        print(f"Predicted for High CBD: {predicted_effects}")

        self.assertEqual(results[0]['effect'], 'Calm')
        print("Success: Correctly distinguished cannabinoid profiles.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_admin_route_sensitivity(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Test if admin route match is prioritized
        self.print_step("Admin Route Sensitivity", "Testing if the same type (Edible vs Tincture) is prioritized.")
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        mock_sessions.return_value = [
            self.create_session(1, 'Edible', 1.0, ['Relaxed'], [{'cannabinoid_id': 1, 'serving': 10.0}], []),
            self.create_session(2, 'Tincture', 1.0, ['Focused'], [{'cannabinoid_id': 1, 'serving': 10.0}], [])
        ]
        knn = KNN(user_id=1)
        
        #Edible
        new_product = {'cannabinoids': [{'cannabinoid_id': 1, 'serving': 10.0}], 'terpenes': [], 'type': 'Edible'}
        results = knn.predict(new_product, 1.0)
        
        predicted_effects = []
        for r in results:
            predicted_effects.append(r['effect'])
        print(f"Predicted for Edible: {predicted_effects}")

        self.assertEqual(results[0]['effect'], 'Relaxed')
        print("Success: Administration route match prioritized.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_no_strong_matches(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Test threshold filtering (similarity > 0.1)
        self.print_step("Threshold Filtering", "Testing that irrelevant history returns no results.")
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        
        #Low dose Tincture
        mock_sessions.return_value = [self.create_session(1, 'Tincture', 0.1, ['Focused'], [{'cannabinoid_id': 1, 'serving': 1.0}], [])]
        knn = KNN(user_id=1)
        
        #High dose Edible with different cannabinoids
        new_product = {
            'cannabinoids': [{'cannabinoid_id': 2, 'serving': 50.0}],
            'terpenes': [{'terpene_id': 3}],
            'type': 'Edible'
        }
        results = knn.predict(new_product, 10.0)
        
        print(f"Results for irrelevant query: {results}")
        self.assertEqual(len(results), 0)
        print("Success: Irrelevant matches filtered out correctly.")

    @patch('recommender.get_all_cannabinoids')
    @patch('recommender.get_all_terpenes')
    @patch('recommender.get_user_sessions')
    def test_empty_history(self, mock_sessions, mock_terpenes, mock_cannabinoids):
        #Test behavior for a brand new user
        self.print_step("Empty History", "Testing that a new user with no sessions gets an empty list gracefully.")
        mock_cannabinoids.return_value = self.mock_cannabinoids
        mock_terpenes.return_value = self.mock_terpenes
        mock_sessions.return_value = []
        
        knn = KNN(user_id=999)
        results = knn.predict({'type': 'Edible', 'cannabinoids': [], 'terpenes': []}, 1.0)
        
        self.assertEqual(results, [])
        print("Success: Empty history handled")

if __name__ == '__main__':
    unittest.main()