function [Pd,PD]=ED_unc(unc,snr_db,Ns)
snr=10.^(snr_db./10);
pf=0.005;k=1000;
for u=1:length(unc)
    un=unc(u);
    for ni=1:length(Ns)
        n=Ns(ni);
        t=linspace(0,1,n);
        x=sin(pi*t);%主用户信号 
        for a=1:length(snr)
            Over_Num=0;%在检测中超过门限的个数
            for b=1:k %进行k次信号检测
                noise=wgn(1,n,0);  %产生功率为1的加性高斯白噪声
                th=un+sqrt(2*un)*qfuncinv(pf)/(sqrt(n));  %计算门限值
                y=sqrt(snr(a))*x+noise;  %CU处接收到的信号
                accum_power(a)=sum(abs(y).^2)/n;  %计算每个CU处接收信号1000个采样点的功率        
                if accum_power(a)>th  %与门限比较进行判决
                    Over_Num=Over_Num+1;
                end
            end
            Pd(a)=Over_Num/k; %不同信噪比下对应的检测概率  
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