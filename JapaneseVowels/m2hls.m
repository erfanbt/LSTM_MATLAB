fileID = fopen('exp.txt','w');
Weights=bfc;
[r,c]=size(Weights);
fprintf(fileID,'={');
for i=1:r
    for j=1:c
        if j==1
            fprintf(fileID,'{');
        end
        if j==c
            if i==r
                fprintf(fileID,'%f}',Weights(i,j));
            else
                fprintf(fileID,'%f},',Weights(i,j));
            end
        else
            fprintf(fileID,'%f,',Weights(i,j));
        end
    end
end
fprintf(fileID,'}');
fclose(fileID);
