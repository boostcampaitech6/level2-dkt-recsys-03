import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--seed', default=42, type=int, help='')
    
    parser.add_argument('--data_dir', default='../../data', type=str, help='')
    parser.add_argument('--model_dir', default='models/', type=str, help='')
    parser.add_argument('--output_dir', default='outputs/', type=str, help='')
    
    parser.add_argument('--run', default='test', type=str, help='')
    
    parser.add_argument('--n_iterations', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    
    parser.add_argument('--num_leaves', default=256, type=int, help='')
    parser.add_argument('--min_data_in_leaf', default=1, type=int, help='')
    parser.add_argument('--max_depth', default=6, type=int, help='')
    parser.add_argument('--early_stopping_round', default=3, type=int, help='')
    parser.add_argument('--max_bin', default=100, type=int, help='')
    
    parser.add_argument('--feats', default=['KnowledgeTag', 'user_correct_answer', 
                                            'user_total_answer','user_acc', 'userID',
                                            'test_mean', 'test_sum', 'test_count', 
                                            'tag_mean','tag_sum','tag_count', 
                                            'assessment_mean', 'assessment_sum','assessment_count',
                                            
                                            ], 
                        type=list, help='')
    
    args = parser.parse_args()
    return args
    