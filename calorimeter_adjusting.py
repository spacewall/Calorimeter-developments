import sys, os
from pprint import pprint
from tkinter.ttk import Frame
from tkinter.filedialog import askopenfilename
from tkinter import PhotoImage, Tk, Canvas, ttk, BOTH, RIDGE, Label, RAISED, NO
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, savgol_filter


class OutputFrame(Frame):
    """Класс отвечает за схему крепления термопар к калориметру с точкой попадания пучка"""
    def __init__(self, window):
        # Сделаем рамку для вывода графиков, рисунков и т. п.
        super().__init__(window)

        # Сделаем заголовок
        self.lable = Label(self)
        self.lable.config(
            text="ИНТЕГРАЛЬНЫЙ КАЛОРИМЕТР",
            relief=RIDGE,
            font=('Artifakt Element', 30),
            borderwidth=4
            )
        self.lable.pack(side='top', expand=True, fill=BOTH)

        # Создадим поле холста
        self.canvas = Canvas(self)
        self.canvas.config(
            bg='#446',
            relief=RAISED,
            width=1000,
            height=840
            )
        self.canvas.pack(side='left', fill=BOTH)

        # Разместим изображение схемы калориметра
        self.img = PhotoImage(file=resource_path('calorimeter_geometry.png'), height=840, width=1100)
        self.canvas.create_image(500, 420, image=self.img, anchor='center')


class AppWindow(Tk):
    """Объект окна Tkinter"""
    # Инициализируем окно
    def __init__(self):
        super().__init__()

        self.title("Юстировка калориметра")
        self.geometry('1600x2560')

        # self.resizable(width=False, height=False)
        self.geometry('1438x1000')


class BlockSocket:
    """Класс отвечает за работу блока управления программой и вывод информации в frame (экземпляр класса OutputFrame)"""
    def __init__(self, window, frame) -> None:
        self.window = window
        self.frame = frame
        # self.data = self.data_loading()

        # Создадим кнопки, заголовки и таблицу блока взаимодействия
        # Добавим заголовок
        _label = Label(
            window,
            text="УПРАВЛЕНИЕ",
            relief=RIDGE,
            font=('Artifakt Element', 30),
            width=25,
            borderwidth=4
            )
        _label.pack(side='top')

        # Зададим стиль текста кнопок в панели управления
        _style = ttk.Style()
        _style.theme_use('classic')
        _style.configure(
            'W.TButton',
            foreground='black',
            font=('Artifakt Element', 20)
            )

        # Добавим кнопку демонстрации точки столкновения
        point_show_button = ttk.Button(
            window,
            text="Показать точку столкновения",
            width=25,
            style='W.TButton',
            command=self.data_analysis
            )
        point_show_button.pack(side='top', ipadx=34, ipady=20)

        # Добавим кнопку смены набора данных
        change_data_pack_button = ttk.Button(
            window,
            text="Выбрать данные",
            width=25,
            style='W.TButton',
            command=self.data_loading
            )
        change_data_pack_button.pack(side='top', ipadx=34, ipady=20)

        # Добавим кнопку построения графиков
        plot_graphs_button = ttk.Button(
            window,
            text="Построить зависимости",
            width=25,
            style='W.TButton',
            command=self.data_plotting
            )
        plot_graphs_button.pack(side='top', ipadx=34, ipady=20)

        # Создадим стиль таблицы
        _style.configure(
            "mystyle.Treeview",
            highlightthickness=0,
            bd=0,
            font=('Artifakt Element', 11)
            )
        _style.configure(
            "mystyle.Treeview.Heading",
            font=('Artifakt Element', 13, 'bold')
            )

        # Добавим таблицу с максимальными значениями температур и энергиями по каждому каналу
        self.columns = ("number", "value", "energy")
        self.tree = ttk.Treeview(
            columns=self.columns,
            show="headings", 
            style='mystyle.Treeview'
            )
        self.tree.pack(fill='y', expand=False, side='left')

        # Определим заголовки
        self.tree.heading("number", text="№")
        self.tree.heading("value", text="T_max, град. C")
        self.tree.heading("energy", text="Q, кДж")

        # Зададим параметры колонок
        self.tree.column("#1", stretch=NO, width=140)
        self.tree.column("#2", stretch=NO, width=182)
        self.tree.column('#3', stretch=NO, width=100)

        self.window.mainloop()


    def table_calculations(self, scale_int) -> int:
        # Поиск точки попадания пучка
        max_points_centered = [max(self.data[:, el] - np.dot(np.ones_like(self.data[:, el]), np.mean(self.data[0:20, el]))) for el in range(0, self.data.shape[1])]
        max_point_number = max_points_centered .index(max(max_points_centered )) + 1 # номер термопары
        max_points = [max(self.data[:, el]) for el in range(0, self.data.shape[1])]

        # Посчитаем энергосодержание потока
        # Найдём точки минимума (список упорядочен по номерам термопар)
        min_points = [min(self.data[:, el]) for el in range(0, self.data.shape[1])]

        # Посчитаем производные
        for collumn in range(0, self.data.shape[1]):
            dU = np.diff(savgol_filter(self.data[:, collumn], 111, 3))
            derivate = list()

            for el in range(1, self.data.shape[0]):
                derivate.append(dU[el - 1]/el)

            if collumn == 0:
                summary = np.zeros(len(derivate))
            else:
                summary =+ np.array(derivate)
        
        # Сделаем срез лишней части массива, где производная неопределена
        summary = summary[100::] * 1000

        # Отыщем минимумы производных
        peaks, _ = find_peaks(-1*summary, height=0.02)

        # Теперь получим номер точки, где установилось термодинамическое равновесие
        balance_indx = peaks[-1] + 100 + scale_int # 100 за компенсацию среза
        # Найдём точку термодинамического равновесия по индексу в массиве
        balance_points = [self.data[balance_indx, el] for el in range(0, self.data.shape[1])]

        # Посчитаем энергосодержание потока при фиксированном интервале
        energies = list()
        for el in range(0, self.data.shape[1]):
            delta = balance_points[el] - min_points[el]
            energies.append(round(delta * 390 * 3 * 0.025, 2)) # 0.390 Дж/(кг °С) * 3 кг * 25 °С/мВ

        # Построим зависимость энергии от выбора точки установления термодинамического равновесия
        interval = range(peaks[-1] + 100, self.data.shape[0])
        energies_matrix = np.ndarray((len(interval), self.data.shape[1]))
        indx = 1
        fig, ax = plt.subplots(figsize=(10, 5))

        for row, balance_indx in enumerate(interval):
            for collumn in range(0, self.data.shape[1]):
                delta = balance_points[collumn] - min_points[collumn]
                energies_matrix[row, collumn] = round(delta * 390 * 3 * 0.025, 2)
            x = self.data[:, collumn]
            ax.plot(x, label=f"Термопара {indx}")
            indx += 1

        plt.show()

        pprint(energies_matrix)
        print(energies_matrix.shape)
        # plt.plot(summary)
        # plt.plot(peaks, summary[peaks], "x")
        # plt.plot(np.zeros_like(summary), "--", color="gray")
        # plt.show()

        # plt.plot(balance_indx * np.ones_like(balance_points), balance_points, "x")
        # data_plotting()

        # Сделаем список из кортежей
        table_data = list()
        indx = 1
        for point in max_points:
            table_data.append((indx, round(point*25, 2), energies[indx - 1])) # умножили на коэффициент перевода в температуру
            indx += 1

        # Очистим содержиое таблицы
        self.tree.delete(*self.tree.get_children())

        # Выведем данные в таблицу
        for el in table_data:
            self.tree.insert("", 'end', values=el)

        return max_point_number

    def thermocouples_location(self, max_point_number) -> None:
        # Построим сетку термопар на схеме
        thermocouples = {
            "1": [False, 650, 515],
            "2": [True, 865, 402],
            "3": [False, 480, 585],
            "4": [True, 655, 402],
            "5": [False, 465, 225],
            "6": [False, 785, 345],
            "7": [False, 645, 290],
            "8": [False, 240, 680],
            "9": [False, 535, 245],
            "10": [True, 780, 402],
            "11": [False, 240, 120],
            "12": [False, 790, 460],
            "13": [True, 475, 402],
            "14": [True, 580, 260]
        }

        for number, coordinates in thermocouples.items():
            is_dashed, x, y = coordinates

            # Зададим условие на цвет точки попадания пучка
            if number == str(max_point_number):
                color = "red"
            else:
                color = "blue"

            # Проверка условия на размещение в переднем/заднем плане
            if is_dashed:
                self.frame.canvas.create_line(x, y, x + 30, y + 30, activefill="green", fill=color, width=5, dash=2)

                self.frame.canvas.create_line(x + 30, y, x, y + 30, activefill="green", fill=color, width=5, dash=2)
            else:
                self.frame.canvas.create_line(x, y, x + 30, y + 30, activefill="green", fill=color, width=5)

                self.frame.canvas.create_line(x + 30, y, x, y + 30, activefill="green", fill=color, width=5)

            # Добавим номера каналов
            if number in ["8", "3", "1", "12", "10"]:
                self.frame.canvas.create_text(x - 25, y + 30, font=('Artifakt Element', 30), text=number, fill="black")
            elif number == "6":
                self.frame.canvas.create_text(x - 25, y + 10, font=('Artifakt Element', 30), text=number, fill="black")
            elif number == "2":
                self.frame.canvas.create_text(x + 55, y - 15, font=('Artifakt Element', 30), text=number, fill="black")
            else:
                self.frame.canvas.create_text(x + 15, y + 45, font=('Artifakt Element', 30), text=number, fill="black")

    def data_analysis(self, scale_int=560) -> None:
        try:
            # Рассчитаем данные для таблицы
            max_point_number = self.table_calculations(scale_int)

            # Выведем в окно сетку термопар
            self.thermocouples_location(max_point_number)
        except AttributeError:
            # Подгрузим данные
            self.data_loading()

            # Рассчитаем данные для таблицы
            max_point_number = self.table_calculations(scale_int)

            # Выведем в окно сетку термопар
            self.thermocouples_location(max_point_number)
        except FileNotFoundError:
            pass

    def data_loading(self) -> None:
        # Запросим директорию файла с данными
        directory = askopenfilename()

        # Импортируем данные калориметра
        self.data = np.genfromtxt(directory, delimiter=';', skip_header=True, skip_footer=True)

    def plots(self) -> None:
        # Построим графики по всем каналам 
        indx = 1
        colors = ['black', 'red', 'blue', 'yellow', 'green', '#8A2BE2', 'pink', 'violet', '#FF4500', '#C71585', '#FF8C00', '#BDB76B', '#4B0082', '#00FF00', '#00FFFF']
        # Закроем все графические окна с зависимостями
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 5))

        for collumn in range(0, self.data.shape[1]):
            x = self.data[:, collumn]
            # peaks, _ = find_peaks(x, height=0.2, width=100, prominence=1)
            # plt.plot(peaks, x[peaks], "x")
            # fig.subplots_adjust(bottom=0.15, left=0.2)
            ax.plot(x, label=f"Термопара {indx}", color=colors[indx - 1])
            indx += 1

        ax.set_xlabel("Время, с")
        ax.set_ylabel("Напряжение, В")
        ax.plot(np.zeros_like(x), "--", color="gray")

        # Выведем графики в окно
        ax.legend()
        ax.set_title("Результаты измерений")
        fig.show()

    def data_plotting(self) -> None:
        try:
            # Выведем зависимости в графическое окно
            self.plots()
        except AttributeError:
            # Подгрузим данные
            self.data_loading()

            # Выведем зависимости в графическое окно
            self.plots()
        except FileNotFoundError:
            pass

def resource_path(relative_path):
    """Возвращает обсолютный путь объекта, работает для PyInstaller при компиляции в один файл"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main():
    """Запуск программы"""
    # Инициализация рабочей области
    window = AppWindow()

    # Инициализация окна вывода
    frame = OutputFrame(window)
    frame.config(borderwidth=4, relief=RIDGE)
    frame.pack(side='left', fill=BOTH, expand=True)

    # Инициализация блока управления
    BlockSocket(window, frame)

if __name__ == "__main__":
    main()