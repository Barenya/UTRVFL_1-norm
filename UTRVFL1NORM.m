function [results,time]=UTRVFL1NORM(train,test,L,c1,c3,w_vec,b_vec,e,Uni)
%w_vec is weight vector
    %b_vec is bias vector
    %c1 is penalty parameter 
    %c3 is penalty parameter for universum data
    %L is hidden layer nodes
    %e is unit vector
    %Uni is universum data

[no_input,no_col]=size(train);
    obs = train(:,no_col); 
   
    x1 = train(:,1:no_col-1);
    y1 = train(:,no_col);
    [no_test,no_col] = size(test);
    xtest = test(:,1:no_col-1);
    ytest = test(:,no_col);
    xtest0 = xtest;

    train1 = [];
    train2 = [];
    
    for i = 1:no_input
        if(obs(i) == 1)
            train1 = [train1;x1(i,1:no_col-1)];
        else
            train2 = [train2;x1(i,1:no_col-1)];
        end;
    end;
    x1 = [train1 ; train2];   %matrix of 

    c2 = c1;
    c4=10^-4;
	%c5 = c4;
    c3=c3
    ep = 0.00001;
    
    [m3,n] = size(Uni);
     e3 = ones(m3,1);
   [m1,n_att] = size(train1);
    U = zeros(m1,L); 
    tic
     for i=1:m1
        for j=1:L 
            prod = train1(i,:) * w_vec(:,j) + b_vec(j);
            U(i,j) = U(i,j) + 1.0 / ( 1.0 + exp(-prod) );    

        end
    end
    U=[U train1];
    [m2,n_att] = size(train2);
    V = zeros(m2,L);
    m4=m1+m2
    for i=1:m2
        for j=1:L
            prod = train2(i,:) * w_vec(:,j) + b_vec(j);
            V(i,j) = V(i,j) + 1.0 / ( 1.0 + exp(-prod) );   

        end
    end
	V=[V train2];
    [m,n_att] = size(x1);
    e1 = ones(m1,1); 
    e2 = ones(m2,1);
    
     [m3,n_att] = size(Uni);
    Uni1 = zeros(m3,L);  
    for i=1:m3
        for j=1:L
            prod = Uni(i,:) * w_vec(:,j) + b_vec(j); 
            Uni1(i,j) = Uni1(i,j) + 1.0 / (1.0 + exp(-prod) );   
        end
    end
	
	O=[Uni1 Uni];
    
    lowb1=zeros(m2+m3,1);
    lowb2=zeros(m1+m3,1);
    upb1 = [c1*e2;c3*e3];
    upb2 = [c2*e1;c3*e3];
    
    H=U;
    G=V;
    HTH = H' * H;
    
    invHTH = inv(HTH + c4 * speye(L+n_att) );
    GO=[G;-O];
    GOINVGOT = GO * invHTH * GO';
   
    GTG = G' * G;
    invGTG = inv (GTG + c4 * speye(L+n_att));
    HO=[H;-O];
    HOINVHOT = HO * invGTG * HO';
   
    f1 = -[e2;(ep-1)*e3]';
    f2 = -[e1;(ep-1)*e3]';
    
    GOINVGOT=(GOINVGOT+GOINVGOT')*0.5;
    HOINVHOT=(HOINVHOT+HOINVHOT')*0.5;
    
    u1 = quadprog(GOINVGOT,f1,[],[],[],[],lowb1,upb1);
    u2 = quadprog(HOINVHOT,f2,[],[],[],[],lowb2,upb2);
    
    time= toc 
    beta1 = invHTH * (O'*u1(m2+1:m2+m3)-G'*u1(1:m2));
    beta2 =  -invGTG * (O'*u2(m1+1:m1+m3)-H'*u2(1:m1));
    
    [n_att,L] = size(w_vec);
    [no_test_input,n_att] = size(xtest0);
    HT = zeros(no_test_input,L);
    for i=1:no_test_input
        for j=1:L
            prod = xtest0(i,:) * w_vec(:,j) + b_vec(j);
            HT(i,j)=HT(i,j) + 1.0 / ( 1.0 + exp(-prod) );    
        end
    end
   HT=[HT xtest0];
   
    ytest1 = HT * beta1;
    ytest2 = HT * beta2;
    
    for i = 1 : size(ytest1,1)
        if abs(ytest1(i)) < abs(ytest2(i))
            classifier(i) = 1;
        else
            classifier(i) = 0;
        end;
    end;
%-----------------------------
match = 0.; 
classifier = classifier';

for i = 1:size(ytest1,1)
    if(classifier(i) == ytest(i))
        match = match+1;
    end;
end;
confmat=confusionmat(y,ytest,'order',[1,0,-1]);
TP=confmat(1,1);
TN=confmat(2,2);
FP=confmat(2,1);
FN=confmat(1,2);

TPR=TP/(TP+FN);
FPR=FP/(FP+TN);

accuracy=(TP+TN)/(TP+TN+FP+FN)*100;
AUC=((1+TPR-FPR)/2)*100;
recall=TP/(TP+FN);
precision=TP/(TP+FP);
f1=2*(precision*recall)/(precision+recall);
gmean=sqrt(precision*recall);
MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
r=[accuracy;AUC;recall;precision;f1;gmean;MC]    
    
