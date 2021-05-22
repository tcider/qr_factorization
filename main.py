# Мы решаем задачу обучения нейрона с использованием псевдообратной матрицы
# Этот метод используется в задаче Регрессии - поэтому будем работать с матрицей обьекты-признакми matrix
# У нас один признак, регрессию к-го мы делаем и множестов обьектов посредтством нахождения коэфициентов w

# Из Numpy используем только Три функции - Транспонирование матрицы, Матричное произведение, Копирование матрицы
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Гиперпараметры модели
# Колличество обьектов в модели
TRAIN_SIZE = 30
# Общее множество обьектов - необходимо для случайный выборки из него TRAIN_SIZE обьектов
POOL_SIZE = 500
# Стандартное отклонения для случаных генераторов
PRECISION = 0.1
# Тк мы будем исследовать нелинейное распределение, задаем степень полиномов используемых для Регрессии
POLY_N = 11


# Допустим наши данные подчинаются распределениею ф-ии cos() - это может быть любая ф-я
def real_function(x):
    y = np.cos(x)
    return y

# Матрица проекции - нужна для факторизации матрицы
def proj(u, a):
    return (np.dot(a,u) / np.dot(u,u)) * u

# Нахождение ортогональной матрицы методом Грама―Шмидта (QR разложение)
def qr_factorization(X):
    X = 1.0 * X
    U = np.copy(X)

    for i in iter(range(1, X.shape[1])):
        for j in iter(range(i)):
            U[:,i] -= proj(U[:,j], X[:,i])

    # Нормализация матрицы
    norm = (U ** 2).sum(axis = 0) ** 0.5
    N = U / norm
    return N

# Используем класс для того чтобы сохранять результаты обучения и применять модель на практике
class Model:
    # Словарь для исходных данных
    base = {}
    # Вектор искомых коэффициентов в процессе обучения
    w = []
    # Массив для результатирующих Y
    result_y = []
    # Скалярное значение для хранения ошибки (по формуле MSE)
    error = 0

    def __init__(self, train_size, pool_size, precision):
        # При инициализации класса создаем случайные вектора X и Y, результат сохраняем в словаре
        result = {}
        # Тк в качестве закономерности мы выбрали периодическую ф-ю, зададим интервал x от 0 до 2Пи
        pool = np.linspace(0, 2 * np.pi, num=pool_size)
        real_y = real_function(pool)
        # Делаем TRAIN выборку x из всего множества х
        result['train_x'] = np.sort(np.random.choice(pool, size=train_size, replace=True))
        # "Придумываем" для вектора х значения, берем реальную ф-ю и добавляем к ней случайную ошибкуш
        result['train_y'] = real_function(result['train_x']) + np.random.normal(0, precision, size=result['train_x'].shape[0])
        self.base = result

    # Непосредственно метод обучения в классе
    def learn(self, poly_n):
        # тк Range Не включает последний
        poly_n += 1
        # Создаем вспомогательный массив единиц
        target_ones = [np.ones(self.base['train_x'].shape[0])]
        # Добавляем вектор с апроксимирующими степенями x
        target = target_ones + list(map(lambda n: self.base['train_x'] ** n, range(1, poly_n)))
        # Получем исходную матрицу Обьекты-Признак
        target_array = np.array(target)
        # Транспонируем
        X = target_array.T
        # Факторизуем X по методу QR с помощью "самодельной" функции
        # Либо можно использовать библиотечную q, r = np.linalg.qr(X)
        q = qr_factorization(X)
        q_t = q.T
        # Искомые обучением коэф-ты равны
        self.w = np.dot(q, q_t)
        # Находим Y по данным коэф-там
        self.result_y = np.dot(self.w, self.base['train_y'])

    def error(self):
        size = len(self.result_y)
        diff = self.result_y - self.base['train_y']
        diff = np.abs(diff)
        diff = diff ** 2
        diff_sum = diff.sum() / size
        self.error = np.round(diff_sum, 2)

    # Визуализация результата
    def visualize(self, precision, poly_n):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = 15, 10
        plt.scatter(self.base['train_x'], self.base['train_y'], label='Данные')
        plt.plot(self.base['train_x'], self.result_y, 'r', label='Результат')
        plt.xlim(self.base['train_x'].min() - precision, self.base['train_x'].max() + precision)
        plt.ylim(self.base['train_y'].min() - precision, self.base['train_y'].max() + precision)
        plt.legend(loc='upper right', prop={'size': 15})
        plt.title(f'Обучение модели регресси полиномом ({poly_n} степени), методом нахождения псевдообратной матрицы (Ошибка по MSE = {self.error})', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def main():
    # Инициализируем генератор случаных чисел
    np.random.seed(2021)
    # Инициализируем класс модели
    our_model = Model(TRAIN_SIZE, POOL_SIZE, PRECISION)
    # Обучаем модель
    our_model.learn(POLY_N)
    # Считаем ошибку по MSE
    our_model.error()
    # Строим график
    our_model.visualize(PRECISION, POLY_N)


if __name__ == "__main__":
    main()



