# Simulating Ant Movement

## Similar Size K Means Clustering Algorithm

In order to find the ant locations in each frame, I clustered on points of certain grey intensities. I found that the K Means algorithm often produced clusters with centers that did not correspond to apparent ant locations. I imagine that this is a result of the nature of the initialization of the starting k points, where depending on the proximity of the closest points to dense groupings of ants, it could be that the cluster centers converge with multiple ants in one cluster.

There is a known Same Size K Means algorithm that works to combat this already, but this would not work for me. The 20 ants each have different amount of pixels that they consume in each photo and searching for clusters of the same size wouldn't produce the results that I wanted. This algorithm isn't native to Scala and I didn't end up implementing it, so I am not able to compare it in application.

This led me to developing a [Similar Size K Means](https://github.com/eherbert/SimulatingAntMovement/blob/master/src/main/scala/utility/SimilarSizeKMeans.scala) clustering algorithm. It works by running a normal K Means and then checking to see if the clusters are evenly weighted. If they are, it returns the cluster centers. If they are not, it moves the cluster centers from the smallest sized clusters to random points from the largest sized clusters, effectively splitting them. You pass the algorithm a stdDevBreakpoint and a stdDevTolerance where the clusters are deemed evenly weighted if the standard deviation is below the stdDevBreakpoint and where, if they are not evenly weighted, the clusters larger than the standard deviation * stdDevTolerance are targeted for splitting. Normal K Means is run again and this process repeats until the converged cluster centers from the K Means algorithm are evenly weighted.

Here are the results of K Means compared to results of Similar Size K Means.

K Means:

![alt text](images/kmeans.png "K Means Results")

Similar Size K Means:

![alt text](images/similarsizekmeans.png "Similar Size K Means Results")