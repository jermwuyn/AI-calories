import unittest
from scripts.audit_nutrition5k import summarize_metadata


class NutritionMetadataTests(unittest.TestCase):
    def test_csv_column_order_and_split_accounting(self):
        files = {
            'nutrition5k_dataset/metadata/dish_metadata_cafe1.csv': (
                b'dish_1,10,20,1,2,3,ingr_0000000001,food,20,10,1,2,3\n'
                b'dish_2,0,5,0,0,0,ingr_0000000002,food,5,7,1,1,1\n'),
            'nutrition5k_dataset/dish_ids/dish_ids_all.txt': b'dish_1\ndish_2\n',
            'nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt': b'dish_1\n',
            'nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt': b'dish_2\n',
            'nutrition5k_dataset/dish_ids/splits/depth_train_ids.txt': b'dish_1\n',
            'nutrition5k_dataset/dish_ids/splits/depth_test_ids.txt': b'',
        }
        report, ids, splits = summarize_metadata(files)
        self.assertEqual(ids, {'dish_1','dish_2'})
        self.assertEqual(report['rgb_splits']['train_test_overlap'],0)
        self.assertEqual(report['zero_total_with_positive_ingredient_sum']['calories_kcal'],1)
        self.assertEqual(report['zero_values']['calories_kcal'],1)
        self.assertEqual(report['invalid_values'],{})
        self.assertEqual(report['depth_split_not_subset_of_rgb_split']['train'],0)

    def test_duplicate_and_bad_ingredient_structure_are_reported(self):
        files={'metadata/dish_metadata_cafe1.csv':b'dish_1,1,2,3,4,5,bad\ndish_1,1,2,3,4,5,bad\n'}
        report,_,_=summarize_metadata(files)
        self.assertEqual(report['invalid_values']['duplicate_dish_id'],1)
        self.assertEqual(report['invalid_values']['ingredient_tuple_structure'],2)


if __name__=='__main__':unittest.main()
