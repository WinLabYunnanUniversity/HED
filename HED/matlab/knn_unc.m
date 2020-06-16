function [Pd,PD]=knn_unc(unc,snr_db,Ns)
snr=10.^(snr_db./10);
k=1000;
%------------------1.��ȷ���ȣ�
for u=1:length(unc)
    unc1=unc(u);
    unc_qujian=linspace(1/unc1,unc1,10);
    index=randperm(length(unc_qujian),1);
    uncertainty=unc_qujian(index);
    %-----------------2.����ʱ�䣨������:
    for s=1:length(Ns)
        n=Ns(s);
        t=linspace(0,1,n);
        x=sin(pi*t);%���û��ź� 
        %-------------3.����ȣ�
        for i=1:length(snr)
            train=[];test=[];train_label=[];test_label=[];
            %-------------4.����k���źż�� 1000
%             for b=1:k %
%             %====================-�����������ź�-=======================
%                 noise=wgn(1,n,0);  %��������Ϊ1�ļ��Ը�˹������
%                 y=sqrt(snr(i))*x+noise;  %ED:���û����յ����ź�
%                 Noise=[];Pn=[];Y=[];
%                 Noise(b,:)=noise.*uncertainty;
%                 Pn(b,:)=sum(abs(noise).^2,1)/n;
%                 Y(b,:)=sqrt(snr(i))*x+noise.*uncertainty;
%             end
            Noise=uncertainty.*wgn(1000,n,0);
            y=sqrt(snr(i))*x+Noise; 
            %-----------�����ź���������������------------
            noise2=Noise;
            PY=sum(abs(y).^2)/n;
            aa=sum(abs(noise2).^2,1)/n; %��������
            %-------------����ѵ�����Ͳ��Լ�--------------
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