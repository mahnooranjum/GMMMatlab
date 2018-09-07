%The following is a detailed coding of the multivariate gaussians 
%Presented by Mahnoor Anjum for Dr. Hassan Aqeel Khan
%the references will be detailed in the end
%============================================================
%I will be starting with multivariate gaussians through mvnrnd() 
%Let us create the mean and sigma matrixes as specified in the assignment
u1 = [0, 0];
u2 = [3,3];
u3 = [0,4];
sigma1 = [1,0.4;0.4,1];
sigma2 = [1,0;0,2];
sigma3 = [0.4,0;0,0.1];
weights = [1/3, 1/3,1/3];
dist1 = mvnrnd(u1,sigma1,333);
dist2 = mvnrnd(u2,sigma2,333);
dist3 = mvnrnd(u3,sigma3,333);
%The following is the result of three separate gaussians
finaldist = [dist1;dist2;dist3];
subplot(1,2,1)
plot(finaldist(:,1), finaldist(:,2), 'o');
%let's create the same distribution with a better function in the following %steps:
combinedu = [u1;u2;u3];
combinedsigma=cat(3,sigma1,sigma2,sigma3);
gmmodel = gmdistribution(combinedu, combinedsigma, weights);
X = random(gmmodel,1000);
subplot(1,2,2)
plot(X(:,1),X(:,2), 'ro');
%Without the Kmeans algorithm, we get the following results: 
obj = gmdistribution.fit(X,3);

%Now we estimate the parameters with Kmeans. 
[prior mu sigma]=EM_init_kmeans(X.', 3)

s = struct('mu',mu.','Sigma',sigma,'PComponents',prior);
options = statset('Display','final');
e =1e-5; 
GMModel=gmdistribution.fit(X,3,'CovType','full','Options',options,'Start',s,'Regularize',e);

