function [A, Ah, AhA] = forward_operator(maps, kmask)
% Create forward, adjoint, and composition operators for MRI reconstruction
%
% Input:
%   maps  - Sensitivity maps [N1 x N2 x Nt]
%   kmask - k-space sampling mask [N1 x N2 x Nc x Nt]
%
% Output:
%   A   - Forward operator (images to multichannel k-space)
%   Ah  - Adjoint operator (k-space to images)
%   AhA - Composition operator (A'*A)
%
% Note: Uses dynamic string evaluation to handle multiple time frames
% 
% Rodrigo A. Lobos, October 2025

[N1, N2, Nc, Nt] = size(kmask);      % N1 x N2 : image dimensions
                                    % Nc      : number of coils
                                    % Nt      : number of time frames

size_sc = N1*N2;
size_mc = N1*N2*Nc;


% Forward operator

st = ['A=@(x) [ '];
for i = 1:Nt
st = [st, 'utils.vect(kmask(:,:,:,' int2str(i) ').*fftshift(fft2(ifftshift(maps.*repmat(reshape(x((' int2str(i) '-1)*size_sc + 1 : ' int2str(i) '*size_sc), [N1, N2]), [1 1 Nc])))));'];
end
st = [st(1:end-1),'];'];
eval(st); % created adjoint data-consistency operator

% Adjoint operator

st = ['Ah=@(x) [ '];
for i = 1:Nt
st = [st, 'utils.vect(sum(conj(maps).*fftshift(ifft2(ifftshift(kmask(:,:,:,' int2str(i) ').*reshape(x((' int2str(i) '-1)*size_mc + 1 : ' int2str(i) '*size_mc), [N1,N2,Nc])))),3));'];
end
st = [st(1:end-1),'];'];
eval(st); % created adjoint data-consistency operator

% Composition of both operators

st = ['AhA=@(x) [ '];
for i = 1:Nt
st = [st, 'utils.vect(sum(conj(maps).*fftshift(ifft2(ifftshift(kmask(:,:,:,' int2str(i) ').*fftshift(fft2(ifftshift(maps.*repmat(reshape(x((' int2str(i) '-1)*size_sc + 1 : ' int2str(i) '*size_sc), [N1, N2]), [1 1 Nc]))))))),3));'];
end
st = [st(1:end-1),'];'];
eval(st); % created adjoint data-consistency operator

end