function x = func_svd_min_eigvec(A)
    temp = A'*A;
    [~,~,V] = svd(temp);    % V is transposed and ordered by eigval already
    x = V(:,end);
end