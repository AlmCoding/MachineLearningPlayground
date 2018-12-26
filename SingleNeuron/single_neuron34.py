import numpy as np
import single_neuron12 as sn


def extend_data1d(x, p=1):
    x_ext = x
    for i in range(p-1):
        if DATA_SET == 1:
            end = x_ext[0, :] * x_ext[-1, :]
        elif DATA_SET == 2:
            m = np.max(x)
            end = np.sin(2*np.pi*(i+1)*x/m)
        x_ext = np.vstack((x_ext, end))
    return x_ext


def predict_y2(x):
    # Normalize Input Samples
    x = extend_data1d(x, P)
    mx = np.tile(x_D_ext_mean, (1, x.shape[1]))
    x = (x - mx) / x_D_ext_stdd
    ones = np.ones((1, x.shape[1]))
    x = np.vstack((ones, x))
    a = w.transpose().dot(x)
    return a


if __name__ == '__main__':
    ########################
    DATA_SET = 2
    ########################
    if DATA_SET == 1:
        import dataset1_linreg as ds
    else:
        import dataset2_linreg as ds
    P = 4
    y_D, x_D = ds.DataSet.get_data()
    x_D_ext = extend_data1d(x_D, P)
    x_D_ext_mean, x_D_ext_stdd = sn.get_norm_params(x_D_ext)
    print(x_D_ext_mean)
    print(x_D_ext_stdd)

    w = np.array([0.1] * (P + 1)).reshape(-1, 1)
    ds.DataSet.plot_model(predict_y2)
    print('Initial Cost: %f' % sn.l2_cost(x_D, y_D, predict_y2))
    grad = sn.gradient_w(x_D_ext, y_D, predict_y2)
    print("Initial gradient:\n{}".format(grad))

    for i in range(20000):
        w = w - 0.01 * sn.gradient_w(x_D_ext, y_D, predict_y2)

    ds.DataSet.plot_model(predict_y2)
    print('Cost: %f' % sn.l2_cost(x_D, y_D, predict_y2))
