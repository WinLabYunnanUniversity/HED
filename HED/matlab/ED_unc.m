function [Pd,PD]=ED_unc(unc,snr_db,Ns)
snr=10.^(snr_db./10);
pf=0.005;k=1000;
for u=1:length(unc)
    un=unc(u);
    for ni=1:length(Ns)
        n=Ns(ni);
        t=linspace(0,1,n);
        x=sin(pi*t);%���û��ź� 
        for a=1:length(snr)
            Over_Num=0;%�ڼ���г������޵ĸ���
            for b=1:k %����k���źż��
                noise=wgn(1,n,0);  %��������Ϊ1�ļ��Ը�˹������
                th=un+sqrt(2*un)*qfuncinv(pf)/(sqrt(n));  %��������ֵ
                y=sqrt(snr(a))*x+noise;  %CU�����յ����ź�
                accum_power(a)=sum(abs(y).^2)/n;  %����ÿ��CU�������ź�1000��������Ĺ���        
                if accum_power(a)>th  %�����ޱȽϽ����о�
                    Over_Num=Over_Num+1;
                end
            end
            Pd(a)=Over_Num/k; %��ͬ������¶�Ӧ�ļ�����  
        end
%         PD(ni,:)=Pd;
%         plot(snr_db,Pd,'-o');hold on;
    end
    PD(u,:)=Pd;
end
% grid minor;
% % legend('x=0dB','x=0.01dB','x=0.1dB','x=0.5dB','x=1dB','x=3dB')
% xlabel('SNR');
% ylabel('pd');