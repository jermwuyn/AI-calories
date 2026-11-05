import unittest
from scripts.build_taxonomy_v2 import make_map


class TaxonomyV2Tests(unittest.TestCase):
    def setUp(self):
        self.mapping={'classes':[{'source_name':s} for s in ('a','b','c','d')],
                      'sources':['30VNFoods','UEC-Food256','VinaFood21','Vietnamese_StreetFood_14Class'],
                      'provenance':[{'source':'30VNFoods','source_class_id':str(i),'source_name':s,'canonical_id':i}
                                    for i,s in enumerate(('a','b','c','d'))]}

    def test_id_stable_when_unrelated_classes_merge(self):
        v2=make_map(self.mapping,[(0,1,'MERGE','','')])
        self.assertEqual(v2['v1_to_v2']['0']['stable_id'],v2['v1_to_v2']['1']['stable_id'])
        self.assertEqual(v2['v1_to_v2']['3']['stable_id'],'food-v2-003')
        self.assertEqual(v2['v1_to_v2']['3']['index'],2)
        self.assertEqual([c['index'] for c in v2['classes']],list(range(3)))
        self.assertEqual(v2['classes'][0]['v1_ids'],[0,1])
        self.assertIsNone(v2['classes'][0]['name_en'])

    def test_nonmerge_cannot_be_violated_transitively(self):
        decisions=[(0,1,'MERGE','',''),(1,2,'MERGE','',''),(0,2,'KEEP_SEPARATE','','')]
        with self.assertRaisesRegex(ValueError,'Contradictory'): make_map(self.mapping,decisions)

    def test_needs_review_never_merges(self):
        v2=make_map(self.mapping,[(0,1,'NEEDS_REVIEW','','')])
        self.assertEqual(v2['class_count'],4)
        self.assertNotEqual(v2['v1_to_v2']['0']['stable_id'],v2['v1_to_v2']['1']['stable_id'])


if __name__=='__main__':unittest.main()
