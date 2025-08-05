def ConvTranspose2D(data: list[list], kernel: list[list], stride: int = 2, padding: int = 0, fill: int = 0, output_padding: int = 0):
    """applies 2D convolution on image

    Args:
        data (list[list]): the 2D matrix which conv is applied on
        kernel (list[list]): the kernel to apply
        stride (int, optional): step size for kernel. Defaults to 1.
        padding (int, optional): padding to add to the data matrix. Defaults to 0.
        fill (int, optional): the number which is used while padding the data. Defaults to 0.

    Raises:
        ValueError: if padding < 0 
        
    Note:
        in this code, Hight is represented ad l or line and width is represented as r or row

    Returns:
        _type_: a feature map 
            new w, h would be:
            new_h = floor((h - kernel_h + 2 * padding) / stride) + 1
            new_w = floor((w - kernel_w + 2 * padding) / stride) + 1
    """
    
    original_l = len(data)
    original_r = len(data[0])
    
    new_l = (original_l - 1) * (stride - 1) + original_l
    new_r = (original_r - 1) * (stride - 1) + original_r
        
    temp = [[0 for __ in range(new_r)] for _ in range(new_l)]
    
    a = 0
    for i in range(original_l):
        b = 0
        for j in range(original_r):
            temp[i+a][j+b] = data[i][j]
            b += stride - 1
        a += stride - 1
            
            
    kl = len(kernel)
    kr = len(kernel[0])
    
    
    # adding padding if padding
    if padding < 0:
        raise ValueError(f"padding cannot be smaller than 0 but got {padding}")
    
    elif padding > 0:
        padded_r = new_r + 2 * padding
        
        for line in temp:
            for _ in range(padding):
                line.insert(0, fill)  # fill the first of the line
            for _ in range(padding):
                line.append(fill)  # fill the end of the line

            
        for _ in range(padding):
            temp.insert(0, [fill for _ in range(padded_r)])  # a line at the top of matrix
            temp.append([fill for _ in range(padded_r)])  # a line at the bottom of matrix
        
    fmap_lines = (new_l - kl + 2 * padding) + 1
    fmap_rows = (new_r - kr + 2 * padding) + 1
    
    feature_map = [[0 for __ in range(fmap_rows)] for _ in range(fmap_lines)]
    
    new_l = len(temp)
    new_r = len(temp[0])
        
    # stride is always 1 cause we already did the zero-inserting
    for i in range(new_l - kl + 1):  # i corresponds to line
        for j in range(new_r - kr + 1):  # j corresponds to row
            sum_ = 0
            for m in range(kl):
                for n in range(kr):
                    sum_ += temp[i+m][j+n] * kernel[m][n]
            
            feature_map[i][j] = sum_
                
    return feature_map


if __name__ == "__main__":
    kernel = [[0, 1, 0],
              [1, 4, 1],
              [0, 1, 0]]

    
    m = [[1,1,1,1],
         [2,2,2,2],
         [1,1,1,1],
         ]
    r = ConvTranspose2D(m, kernel, stride=2, padding=1)
