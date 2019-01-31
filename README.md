Previously implemented an eigenface recognition by hand, now ready to use the 
scikit-learn library for achieving the same result. Biggest takeaway is that the library
took care of the most tedious part of the implementation, namely writing functions to:
            - splitting the data
            - extract eigenvectors
            - construct facespace
            - project testing faces back to facespace

previously, the hand-written implementation was heavy on the PCA and not so much on the
actual classification. was able to explore this with the SVC from sklearn this time. 
Also good introduction to the gridsearch for parameter optimization and a more robust
definition of recall and precision for quantifying performance.

BUT, haven't figured out yet how to do outlier detection yet. This current code operates
under the assumption that every test face belongs to a known subject in the database. It 
doesn't address the possibility that the test face is a new face that belongs to no known
subject, or the possibility that the test 'face' is not a face at all. This is something
that the original eigenface algorithm was very good at by using the distance to the facespace
as a direct metric. 