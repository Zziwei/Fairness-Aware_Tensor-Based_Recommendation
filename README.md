# Fairness-Aware_Tensor-Based_Recommendation
This is the implementation of our paper: 

Ziwei Zhu, Xia Hu, and James Caverlee. 2018. Fairness-Aware Tensor-Based Recommendation. In The 27th ACM International Conference on Information and Knowledge Management (CIKM ’18), October 22–26, 2018, Torino, Italy. 


The implementation adopts the open source project: PyTen (https://github.com/tamu-helios/pyten).

A toy twitter expert dataset is provided: 
1) 'tensor_rating.csv' contains the user-expert-topic-rating data, where there is a '1' at 'rating' column, means the user like the expert w.r.t. the topic;
2) 'tensor_ethnicity.csv' contains the user-expert-topic-ethnicity data, where '0', '1' and '2' at 'ethnicity' column represent the expert ethnicity; '0' - Asian, '1' - White, '2' - African American;
2) 'tensor_gender.csv' contains the user-expert-topic-gender data, where '0' and '1' at 'gender' column represent the expert gender; '0' - male, '1' - female.

Author: Ziwei Zhu (zhuziwei@tamu.edu)
