import sys, os
from pprint import pprint
import tkinter
from tkinter.ttk import Frame
from tkinter.filedialog import askopenfilename
from tkinter import PhotoImage, Tk, Canvas, ttk, BOTH, RIDGE, Label, RAISED, NO, Entry
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


class OutputFrame(Frame):
    """Класс отвечает за графическое окно с схемой крепления термопар к калориметру"""
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
    """Объект окна Tkinter - окно приложения"""
    # Инициализируем окно
    def __init__(self, title: str):
        super().__init__()

        self.title(title)
        # self.geometry("1600x2560")

        # self.resizable(width=False, height=False)
        self.geometry("1438x1000")


class BlockSocket:
    """Класс отвечает за отрисовку блока управления, расчёты и вывод информации"""
    def __init__(self, window, frame) -> None:
        self.window = window
        self.frame = frame

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
        self._style = ttk.Style()
        self._style.theme_use('classic')
        self._style.configure(
            'W.TButton',
            foreground='black',
            font=('Artifakt Element', 19)
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

        # Добавим кнопку построения исходных зависимостей
        plot_graphs_button = ttk.Button(
            window,
            text="Построить исходные зависимости",
            width=25,
            style='W.TButton',
            command=self.data_plotting
            )
        plot_graphs_button.pack(side='top', ipadx=34, ipady=20)

        # Добавим кнопку построения энергетических зависимостей
        plot_energy_graphs_button = ttk.Button(
            window,
            text="Построить энергетические зависимости",
            width=25,
            style='W.TButton',
            command=self.energy_dependencies_plots
            )
        plot_energy_graphs_button.pack(side='top', ipadx=34, ipady=20)

        # Добавим кнопку редактирования исходных данных
        point_show_button = ttk.Button(
            window,
            text="Редактировать данные",
            width=25,
            style='W.TButton',
            command=self.change_input_data_interface
            )
        point_show_button.pack(side='top', ipadx=34, ipady=20)

        # Создадим стиль таблицы
        self._style.configure(
            "mystyle.Treeview",
            highlightthickness=0,
            bd=0,
            font=('Artifakt Element', 11)
            )
        self._style.configure(
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

    def change_input_data_interface(self):
        """
        Выводит графическое окно для редактирования входных данных, за редактирование не отвечает
        """
        self.changer_window = AppWindow("Редактирование данных")
        self.changer_window.geometry("400x200")
        self.changer_window.resizable(width=False, height=False)

        try:
            # Добавим инструкцию
            _label = Label(
                self.changer_window,
                text=f"Введите номера термопар\n через запятую (всего термопар: {self.data.shape[1]})",
                font=('Artifakt Element', 19),
                width=26,
                borderwidth=4
            )

            # Добавим кнопку редактирования исходных данных
            data_change_button = ttk.Button(
                self.changer_window,
                text="Удалить данные",
                width=26,
                style='W.TButton',
                command=self.change_input_data
                )
            
            # Создадим окно ввода текста
            self.user_input = Entry(
                self.changer_window, 
                font=('Artifakt Element', 19),
                width=24,
                background='white',
                foreground='black'
                )
            
            _label.pack(side='top')
            self.user_input.pack(side='top')
            data_change_button.pack(side='top')

            self.changer_window.mainloop()
        except AttributeError:
            self.changer_window.destroy()
            
            self.error_handler(f"Данные не загружены,\n выберите данные и повторите попытку")

    def error_handler(self, string):
        """Выводит сообщение об ошибке в графическое окно"""
        self.changer_window = AppWindow("Ошибка")
        self.changer_window.geometry("400x100")
        self.changer_window.resizable(width=False, height=False)

        # Добавим инструкцию
        _label = Label(
            self.changer_window,
            text=string,
            font=('Artifakt Element', 19),
            width=28,
            borderwidth=4
        )

        _label.pack(side='top')
        
        self.changer_window.mainloop()

    def change_input_data(self):
        """
        Отвечает за редактирование подгруженных массивов self.data
        """
        channels = self.user_input.get().replace(" ", "").split(",")
        
        self.changer_window.destroy()

        if "".join(channels).isnumeric():
            try:
                self.new_data = self.data

                for channel in frozenset(channels):
                    # self.new_data = np.delete(self.new_data, int(channel) - 1, 1)
                    self.data[:, int(channel) - 1] = np.zeros_like(self.data.shape[0])
            except IndexError:
                self.error_handler(f"Вы вышли за рамки от 1 до {self.data.shape[1]}")
        else:
            self.error_handler("Вы ввели не число, повторите попытку")

    def pre_calculations(self, smooth_param=350) -> int:
        """
        Вычисляет точки термодинамического равновесия по исходным данным, возвращает индекс точки попадания пучка
        """
        # Поиск точки попадания пучка
        max_points_centered = [max(self.data[:, el] - np.dot(np.ones_like(self.data[:, el]), np.mean(self.data[0:20, el]))) for el in range(0, self.data.shape[1])]
        max_point_number = max_points_centered.index(max(max_points_centered)) + 1 # номер термопары
        self.max_points = [max(self.data[:, el]) for el in range(self.data.shape[1])]

        # Посчитаем энергосодержание потока
        # Найдём точки минимума (список упорядочен по номерам термопар)
        self.min_points = list()
        for el in range(0, self.data.shape[1]):
            if self.data[- 1, el] <= self.data[0, el]:
                shape = int(self.data.shape[0] / 3)
                self.min_points.append(min(self.data[:, el][0 : shape]))
            else:
                self.min_points.append(min(self.data[:, el]))

        min_point_numbers = [list(self.data[:, el]).index(self.min_points[el]) for el in range(self.data.shape[1])]
        self.min_point_number = max(min_point_numbers)

        # Построим зависимость энергии от выбора точки установления термодинамического равновесия
        SLICE_FIELD = slice(self.min_point_number + 10, self.data.shape[0])
        self.interval_data = self.data[SLICE_FIELD, :]
        energy_packet = np.zeros(self.interval_data.shape)

        for collumn in range(self.interval_data.shape[1]):
            self.delta = self.interval_data[:, collumn] - np.ones_like(self.interval_data[:, collumn]) * self.min_points[collumn]
            energy_packet[:, collumn] = np.dot(self.delta, 390 * 3 * 0.025)

        self.error = list()
        for row in range(self.interval_data.shape[0]):
            max_energy = max(energy_packet[row, :])
            packet = list(energy_packet[row, :])
            while max_energy == 0:
                packet.remove(0)
                max_energy = max(packet)
            error_rate = max_energy - min(energy_packet[row, :])

            self.error.append(error_rate)

        smoothed_error = gaussian_filter1d(self.error, smooth_param)

        error_derivates = np.gradient(np.gradient(smoothed_error))
        self.infls = np.where(np.diff(np.sign(error_derivates)))[0]
        
        return max_point_number

    def extrapolation(self, y, ax, collumn):
        self.pre_calculations()

        start_num = self.min_point_number - 10 + self.infls[0]

        y = gaussian_filter1d(y[start_num:self.data.shape[0]], 20)
        x = np.linspace(start_num, y.shape[0], num=y.shape[0])

        fitting_parameters, covariance = curve_fit(self.cbroot_fit, x, y)
        a, b = fitting_parameters

        next_x = np.linspace(self.min_point_number, start_num, 5)
        next_y = self.cbroot_fit(next_x, a, b)

        ax.plot(next_x, next_y, 'x')
        
    def cbroot_fit(self, t, a, b):
        return a * np.cbrt(t) + b

    def table_calculations(self):
        """Вычисляет энергии и заполняет таблицу в меню"""
        # Блок анализа производной
        # plt.plot(summary)
        # x = self.data[:, collumn]
        # plt.plot(peaks, x[peaks], "x")
        # plt.plot(np.zeros_like(x), "--", color="gray")
        # plt.show()

        # Теперь получим номер точки, где установилось термодинамическое равновесие
        # balance_indx = self.peaks[-1] + 100 + self.infls[0] # 100 за компенсацию среза
        balance_indx = self.min_point_number + self.infls[0] - 10
        # Найдём точку термодинамического равновесия по индексу в массиве
        balance_points = [self.data[balance_indx, el] for el in range(0, self.data.shape[1])]

        # Посчитаем энергосодержание потока при фиксированном интервале
        energies = list()
        for el in range(0, self.data.shape[1]):
            delta = balance_points[el] - self.min_points[el]
            energies.append(round(delta * 390 * 3 * 0.025, 2)) # 0.390 Дж/(кг °С) * 3 кг * 25 °С/мВ

        # Сделаем список из кортежей
        table_data = list()
        for indx, point in enumerate(self.max_points):
            # Добавим условие на удалённые каналы
            if round(point*25) == 0:
                table_data.append((indx + 1, "-", "-"))
            else:
                table_data.append((indx + 1, round(point*25, 2), energies[indx])) # умножили на коэффициент перевода в температуру

        # Очистим содержиое таблицы
        self.tree.delete(*self.tree.get_children())

        # Выведем данные в таблицу
        for el in table_data:
            self.tree.insert("", 'end', values=el)
    
    def energy_dependencies_plots(self):
        """Вывод энергетических зависимостей в новое окно"""
        # Получим необходимые переменные
        try:
            self.pre_calculations()
        except AttributeError:
            self.data_loading()
            self.pre_calculations()
        except FileNotFoundError:
            pass        

        graphic_window = AppWindow("Энергетические зависимости")
        plt.close()
        fig, (ax_1, ax_2) = plt.subplots(2, 2, figsize=(12, 9))
        
        for collumn in range(self.interval_data.shape[1]):
            delta = self.interval_data[:, collumn] - np.ones_like(self.interval_data[:, 0]) * self.min_points[collumn]
            # energies = gaussian_filter1d(np.dot(delta, 390 * 3 * 0.025), 20)
            energies = savgol_filter(np.dot(delta, 390 * 3 * 0.025), 150, 5)

            # ax_1[0].plot(savgol_filter(energies, 111, 5), label=f"Термопара {collumn + 1}")

            # Условие на удалённые каналы
            if energies.all(0):
                ax_1[0].plot(energies, label=f"Термопара {collumn + 1}")
                # ax_1[1].plot(np.divide(np.diff(savgol_filter(energies, 550, 5)), np.diff(np.array(range(0, interval_data.shape[0])))))
                # ax_1[1].plot(savgol_filter(np.diff(energies), 550, 5))
                ax_1[1].plot(np.diff(energies), label=f"Термопара {collumn + 1}")

        ax_1[0].plot(np.zeros_like(energies), "--", color="gray")
        ax_1[1].plot(np.zeros_like(energies), "--", color="gray")
        ax_1[0].set_xlabel("Время, с")
        ax_1[1].set_xlabel("Время, с")
        ax_1[0].set_ylabel("Q, кДж")
        ax_1[1].set_ylabel("dQ, кДж")
        ax_1[0].set_title("Энергия Q, переданная калориметру")
        ax_1[1].set_title("Дифференциальная энергия dQ, переданная калориметру")
        ax_1[0].legend()
        ax_1[1].legend()

        for i, infl in enumerate(self.infls, 1):
            ax_2[1].axvline(x=infl, color='red')
            ax_2[0].axvline(x=infl, color='red')
            ax_1[1].axvline(x=infl, color='red')
            ax_1[0].axvline(x=infl, color='red')

        ax_2[0].plot(gaussian_filter1d(self.error, 20))
        ax_2[0].set_title("Погрешность (разность показаний)")
        ax_2[0].set_xlabel("Время, с")
        ax_2[0].set_ylabel("∆Q, кДж")
        
        # for collumn in range(interval_data.shape[1]):
        #     # ax_2[1].errorbar(range(len(energy_packet[:, collumn])), energy_packet[:, collumn], yerr=error, fmt='o-', ecolor='red', capsize=4)
            # ax_2[1].plot(energy_packet[:, collumn] - error)

        ax_2[1].plot(gaussian_filter1d(np.gradient(self.error), 20))
        ax_2[1].plot(np.zeros_like(self.error), "--", color="gray")
        ax_2[1].set_title("Дифференциальная погрешность")
        ax_2[1].set_xlabel("Время, с")
        ax_2[1].set_ylabel("d(∆Q), кДж")

        canvas = FigureCanvasTkAgg(fig, graphic_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=100)

        graphic_window.mainloop()
        
    def thermocouples_location(self, max_point_number) -> None:
        """Строит схему крепления термопар в OutputFrame"""
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

    def data_analysis(self) -> None:
        """
        Вызывается по кнопке, заполняет таблицу и строит схему крепления термопар
        """
        try:
            # Рассчитаем данные для таблицы
            max_point_number = self.pre_calculations()
            self.table_calculations()

            # Выведем в окно сетку термопар
            self.thermocouples_location(max_point_number)
        except AttributeError:
            # Подгрузим данные
            self.data_loading()

            # Рассчитаем данные для таблицы
            max_point_number = self.pre_calculations()
            self.table_calculations()

            # Выведем в окно сетку термопар
            self.thermocouples_location(max_point_number)
        except FileNotFoundError:
            pass

    def data_loading(self) -> None:
        """
        Вызывается другими функциями или по кнопке, запрашивает файл с данными, создаёт аттрибут data
        """
        # Запросим директорию файла с данными
        directory = askopenfilename()

        # Импортируем данные калориметра
        self.data = np.genfromtxt(directory, delimiter=';', skip_header=True, skip_footer=True)

    def plots(self, graphic_window: AppWindow) -> None:
        """
        Функция строит зависимости по исходным данным в отдельном окне Tkinter. Важно: окно не очищается - каждый раз строится новое.
        """

        indx = 1
        colors = ['black', 'red', 'blue', 'yellow', 'green', '#8A2BE2', 'pink', 'violet', '#FF4500', '#C71585', '#FF8C00', '#BDB76B', '#4B0082', '#00FF00', '#00FFFF']

        fig, ax = plt.subplots(figsize=(10, 5))

        for collumn in range(0, self.data.shape[1]):
            y = self.data[:, collumn]
            self.extrapolation(y, ax, collumn)
            # peaks, _ = find_peaks(x, height=0.2, width=100, prominence=1)
            # plt.plot(peaks, x[peaks], "x")
            # fig.subplots_adjust(bottom=0.15, left=0.2)
            ax.plot(y, label=f"Термопара {indx}", color=colors[indx - 1])
            indx += 1

        ax.set_xlabel("Время, с")
        ax.set_ylabel("Напряжение, В")
        ax.plot(np.zeros_like(y), "--", color="gray")

        # Выведем графики в окно
        ax.legend()
        ax.set_title("Результаты измерений")
        
        self.canvas = FigureCanvasTkAgg(fig, graphic_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=100)

        graphic_window.mainloop()

    def data_plotting(self) -> None:
        """Вызывается по кнопке, строит график по исходным данным"""
        figure = AppWindow("Исходные зависимости")

        try:
            # Выведем зависимости в графическое окно
            self.plots(figure)
        except AttributeError:
            # Подгрузим данные
            self.data_loading()

            # Выведем зависимости в графическое окно
            self.plots(figure)
        except FileNotFoundError:
            figure.destroy()


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
    window = AppWindow("Юстировка калориметра")

    # Зададим условия работы графических окон с matplotlib (окна будут закрываться всегда)
    matplotlib.use("Agg"),

    # Инициализация окна вывода
    frame = OutputFrame(window)
    frame.config(borderwidth=4, relief=RIDGE)
    frame.pack(side='left', fill=BOTH, expand=True)

    # Инициализация блока управления
    BlockSocket(window, frame)

if __name__ == "__main__":
    main()