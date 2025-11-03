import numpy as np

from partial import partial_ot_1d


def random_slice(dim):
    theta=np.random.random((100,dim)) #100 random projections
    theta=np.stack([th/np.sqrt((th**2).sum()) for th in theta])
    return theta

def ortho_slice(dim):
    theta = np.eye(dim) #to be sure that all dimensions are considered
    theta = theta +  np.random.rand(theta.shape[0], theta.shape[1]) #add noise
    return theta
    

def all_sliced_pw(X, Y, theta, k=-1):
        '''
        compute PAWL for several lines and return **all** solutions
        '''
        n, dn = X.shape # VD: Có 800 mẫu dlieu, 784 chiều => n = 800, dn = 784
        m, _ = Y.shape # VD: m = 800
        if theta is None:
            # ortho_slice(784) tạo ma trận 784×784
            theta = ortho_slice(dn).T 
            # Thêm 100 random projections => theta.shape = (784, 884)
            theta = np.hstack((theta.T, random_slice(dn).T))
        if k == -1:
            k = min(n, m) # k = 800
        X_line = np.matmul(X, theta)  # (800, 784) × (784, 884) = (800, 884)
        Y_line = np.matmul(Y, theta)  # (800, 784) × (784, 884) = (800, 884)


        cum_ot_lines = np.zeros((theta.shape[1], k)) # (884, 800)
        all_x_lines = np.zeros((theta.shape[1], k)) # (884, 800)
        all_y_lines = np.zeros((theta.shape[1], k)) # (884, 800)

        for i in range(theta.shape[1]): # i in range(884)
            # Chiếu dữ liệu theo hướng i
            x_proj, y_proj = X_line[:,i], Y_line[:,i]
            # Lưu chỉ số sắp xếp tăng dần
            arg_sort_proj_x = np.argsort(x_proj)
            arg_sort_proj_y = np.argsort(y_proj)

            # Danh sách 800 x ghép với 800 y và marginal costs của 800 cặp
            # ind_x.shape = (800,), ind_y.shape = (800,), marg_costs.shape = (800,)
            ind_x, ind_y, marg_costs = partial_ot_1d(x_proj, y_proj, k)
            
            # Tính tổng dồn marginal costs
            cum_ot_lines[i] = np.cumsum(marg_costs)
            
            # Sắp xếp lại chỉ số theo thứ tự tăng dần: 0,1,2,...,799
            sorted_ind_x, sorted_ind_y = np.sort(ind_x), np.sort(ind_y)
            # Lấy dữ liệu theo chỉ số đã sắp xếp
            all_x_lines[i] = arg_sort_proj_x[sorted_ind_x]
            all_y_lines[i] = arg_sort_proj_y[sorted_ind_y]
                
        return all_x_lines, all_y_lines, cum_ot_lines


class SlicedPartialOT:
    def __init__(self, n_proj, max_iter_partial=None) -> None:
        self.n_proj = n_proj
        self.max_iter_partial = max_iter_partial

    def draw_direction(self, d):
        w = np.random.randn(d)
        w /= np.sqrt(np.sum(w ** 2, axis=-1, keepdims=True))
        return w

    def project_in_1d(self, x, y, w):
        w /= np.linalg.norm(w)
        proj_x = np.dot(x, w)
        proj_y = np.dot(y, w)
        return proj_x, proj_y


class MonteCarloSlicedPartialOT(SlicedPartialOT):
    def fit(self, x, y):
        assert x.shape[1] == y.shape[1]
        n, d = x.shape[:2]
        m, d = y.shape[:2]

        selection_freq_x = np.zeros((n, ))
        selection_freq_y = np.zeros((m, ))
        avg_cost = 0.
        for _ in range(self.n_proj):
            w = self.draw_direction(d)
            proj_x, proj_y = self.project_in_1d(x, y, w)
            ind_x, ind_y, marginal_costs = partial_ot_1d(proj_x, proj_y, self.max_iter_partial)
            selection_freq_x[ind_x] += 1.
            selection_freq_y[ind_y] += 1.
            avg_cost += np.sum(marginal_costs) # Sum of marginal costs is the total cost
        
        return selection_freq_x / self.n_proj, selection_freq_y / self.n_proj, avg_cost / self.n_proj


class PartialSWGG(SlicedPartialOT):
    def fit(self, x, y, projections=None):
        assert x.shape[1] == y.shape[1]
        n, d = x.shape[:2]
        m, d = y.shape[:2]

        min_cost = np.inf
        indices_x, indices_y = None, None
        best_w = None
        for i in range(self.n_proj):
            if projections is None:
                w = self.draw_direction(d)
            else:
                w = projections[i]
            proj_x, proj_y = self.project_in_1d(x, y, w)
            ind_x, ind_y, _ = partial_ot_1d(proj_x, proj_y, self.max_iter_partial)

            sorted_ind_x = np.sort(ind_x)
            sorted_ind_y = np.sort(ind_y)
            subset_x = x[sorted_ind_x]
            subset_y = y[sorted_ind_y]
            cost = np.sum(np.abs(subset_x - subset_y))

            if cost < min_cost:
                indices_x = ind_x
                indices_y = ind_y
                min_cost = cost
                best_w = w
        
        return indices_x, indices_y, best_w, min_cost



if __name__ == "__main__":
    # n = 100
    # d = 50
    # n_outliers = 10
    # np.random.seed(0)

    # outliers_x = np.random.choice(n, size=n_outliers, replace=False)
    # ind_outliers_x = np.zeros((n, d))
    # ind_outliers_x[outliers_x] = 3.
    # x = np.random.rand(n, d) + ind_outliers_x

    # outliers_y = np.random.choice(n, size=n_outliers, replace=False)
    # ind_outliers_y = np.zeros((n, d))
    # ind_outliers_y[outliers_y] = 3.
    # y = np.random.rand(n, d) - ind_outliers_y
    
    # sliced = MonteCarloSlicedPartialOT(n_proj=100, max_iter_partial=n-n_outliers)
    # freq_x, freq_y, _ = sliced.fit(x, y)
    # print("x")
    # print(freq_x)
    # print(freq_x[outliers_x])
    # print("y")
    # print(freq_y)
    # print(freq_y[outliers_y])
    
    # sliced = PartialSWGG(n_proj=100, max_iter_partial=n-n_outliers)
    # ind_x, ind_y, theta, min_cost = sliced.fit(x, y)
    # print(min_cost)

# Tạo dữ liệu ví dụ đơn giản
    np.random.seed(42)
    n, d = 6, 2  # 6 điểm, 2 chiều

    # Tập X: 4 điểm bình thường + 2 outliers
    x = np.array([
        [1.0, 1.0],   # điểm bình thường
        [1.2, 0.8],   # điểm bình thường  
        [0.8, 1.2],   # điểm bình thường
        [1.1, 0.9],   # điểm bình thường
        [5.0, 5.0],   # outlier
        [6.0, 4.0]    # outlier
    ])

    # Tập Y: tương tự X nhưng dịch chuyển một chút
    y = np.array([
        [2.0, 2.0],   # điểm bình thường
        [2.2, 1.8],   # điểm bình thường
        [1.8, 2.2],   # điểm bình thường  
        [2.1, 1.9],   # điểm bình thường
        [-3.0, -2.0], # outlier
        [-2.5, -3.0]  # outlier
    ])

    print("Dữ liệu X:")
    print(x)
    print("\nDữ liệu Y:")
    print(y)

    # Sử dụng MonteCarloSlicedPartialOT
    print("\n=== MonteCarloSlicedPartialOT ===")
    mc_sliced = MonteCarloSlicedPartialOT(n_proj=50, max_iter_partial=4)  # chỉ lấy 4/6 điểm tốt nhất
    freq_x, freq_y, avg_cost = mc_sliced.fit(x, y)

    print(f"Tần suất lựa chọn điểm X: {freq_x}")
    print(f"Tần suất lựa chọn điểm Y: {freq_y}")
    print(f"Chi phí trung bình: {avg_cost:.4f}")

    # Điểm nào có tần suất cao thì ít có khả năng là outlier
    print(f"Điểm X có tần suất thấp nhất (có thể là outliers): {np.argsort(freq_x)[:2]}")
    print(f"Điểm Y có tần suất thấp nhất (có thể là outliers): {np.argsort(freq_y)[:2]}")

    # Sử dụng PartialSWGG
    print("\n=== PartialSWGG ===")
    swgg = PartialSWGG(n_proj=20, max_iter_partial=4)
    ind_x, ind_y, best_w, min_cost = swgg.fit(x, y)

    print(f"Chỉ số điểm X được chọn: {np.sort(ind_x)}")
    print(f"Chỉ số điểm Y được chọn: {np.sort(ind_y)}")
    print(f"Hướng projection tốt nhất: {best_w}")
    print(f"Chi phí tối thiểu: {min_cost:.4f}")

    # Hiển thị điểm được chọn
    print(f"Điểm X được chọn:\n{x[np.sort(ind_x)]}")
    print(f"Điểm Y được chọn:\n{y[np.sort(ind_y)]}")