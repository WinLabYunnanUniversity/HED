function [Pd,PD]=knn_unc(unc,snr_db,Ns)
snr=10.^(snr_db./10);
k=1000;
%------------------1.不确定度：
for u=1:length(unc)
    unc1=unc(u);
    unc_qujian=linspace(1/unc1,unc1,10);
    index=randperm(length(unc_qujian),1);
    uncertainty=unc_qujian(index);
    %-----------------2.采样时间（点数）:
    for s=1:length(Ns)
        n=Ns(s);
        t=linspace(0,1,n);
        x=sin(pi*t);%主用户信号 
        %-------------3.信噪比：
        for i=1:length(snr)
            train=[];test=[];train_label=[];test_label=[];
            %-------------4.进行k次信号检测 1000
%             for b=1:k %
%             %====================-产生噪声和信号-=======================
%                 noise=wgn(1,n,0);  %产生功率为1的加性高斯白噪声
%                 y=sqrt(snr(i))*x+noise;  %ED:次用户接收到的信号
%                 Noise=[];Pn=[];Y=[];
%                 Noise(b,:)=noise.*uncertainty;
%                 Pn(b,:)=sum(abs(noise).^2,1)/n;
%                 Y(b,:)=sqrt(snr(i))*x+noise.*uncertainty;
%             end
            Noise=uncertainty.*wgn(1000,n,0);
            y=sqrt(snr(i))*x+Noise; 
            %-----------计算信号能量和噪声能量------------
            noise2=Noise;
            PY=sum(abs(y).^2)/n;
            aa=sum(abs(noise2).^2,1)/n; %噪声能量
            %-------------构建训练集和测试集--------------
            index_PY=randperm(length(PY),length(PY)/2);
            index_PN=randperm(length(aa),length(aa)/2);
            database=[PY,aa];labels=[ones(1,length(PY)),zeros(1,length(aa))];
            index_database=linspace(1,length(database),length(database));
            index_train=randperm(length(database),0.75*length(database));
            index_test=setdiff(index_database,index_train);
            train=database(index_train);train_label=labels(index_train);
            test=database(index_test);test_label=labels(index_test);
            %----------------------KNN----------------------
            K=knnclassify(test',train',train_label',3,'euclidean', 'nearest')';
            TN=0;FP=0;FN=0;TP=0;
             for j=1:length(K)
                    if isequal(K(j),test_label(j),1)  
                           TN=TN+1;
                    else    
                           FP=FP+1; 
                    end
            end 
            Pd(i)=TN/length(test_label(test_label==1));
        end
%         PD(s,:)=Pd;
%         plot(snr_db,Pd,'-o');hold on;
    end 
    PD(u,:)=Pd;
end
% grid minor;
% title('KNN');
% xlabel('SNR');ylabel('Pd');
% legend('x=0dB','x=0.01dB','x=0.1dB','x=0.5dB','x=1dB','x=3dB')
% hold off;