function [out] = dotp_asmult(ii,jj,WL,FL)

T = numerictype('WordLength',WL,'FractionLength',FL);
T.Signed = true;

ii = fi(ii,'numerictype',T);
jj = fi(jj,'numerictype',T);

ii=double(ii)*(2^FL);
jj=double(jj)*(2^FL);

for u=1:length(ii)
    if(ii(u)<0  &&  jj(u)<0)  i=-ii(u)-1; j=-jj(u)-1; sign=0;  end
    if(ii(u)<0  &&  jj(u)>=0)  i=-ii(u)-1; j=jj(u);    sign=1;  end
    if(ii(u)>=0  &&  jj(u)<0)  i=ii(u);    j=-jj(u)-1; sign=1;  end
    if(ii(u)>=0 &&  jj(u)>=0) i=ii(u);    j=jj(u);    sign=0;  end
    
    if ( i==0 )
        m=0;
    else
        k=0;
        while 1
            if ( i>=2^k && i<2^(k+1) )
                if ( ((2^k+2^(k+1))/2) < i)
                    m=2^(k+1);
                else
                    m=2^k ;
                end
                break
            end
            k=k+1;
        end
    end
    
    if ( j==0 )
        n=0;
    else
        l=0;
        while 1
            if( j>=2^l && j<2^(l+1) )
                if ( ((2^l+2^(l+1))/2) < j)
                    n=2^(l+1);
                else
                    n=2^l ;
                end
                break
            end
            l=l+1;
        end
    end
    
    if (sign==0)
        out(u)=m*j+n*i-m*n;
    else
        out(u)=-(m*j+n*i-m*n);
    end
end

out=out/(2^(2*FL));
out = fi(out,'numerictype',T);

end