from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QFileDialog, QStyle, QLabel
from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem, QHeaderView # for tables of result
from PyQt5.QtGui import QIcon, QPalette, QPixmap, QIcon
from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget

import sys, os
from pathlib import Path

import time
# Ubuntu - JiaWen
BASE_DIR = os.getcwd()
while os.path.basename(BASE_DIR) != "fyp_team4c":
    path = Path(BASE_DIR)
    BASE_DIR = str(path.parent)
    if BASE_DIR == '/':
        print("Please call this script in the fyp_team4c directory")
        break
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction')

# Windows - Yi Chang
# BASE_DIR = str(Path(os.getcwd()).parent)
# PREDICTION_DIR = BASE_DIR+'\\prediction'

print(PREDICTION_DIR)
sys.path.append(PREDICTION_DIR)
from run_prediction import run_prediction

# import UI files from QT Designer
# 0_mainUI.ui

class ImgWidget(QLabel):

    def __init__(self, imagePath, parent=None):
        super(ImgWidget, self).__init__(parent)
        pic = QtGui.QPixmap(imagePath)
        self.setPixmap(pic)


class MainUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainUI, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('0_mainUI.ui', self) # Load the .ui file

        ## Backend Variables ###

        # BACKEND ======== Change the value to fit the name of model used
        self.model_selection = {
            "Bayes":"Bayesian",
            "LSTM":"LSTM",
            "ConvLSTM":"CONVLSTM",
        }
        # ConvLSTM as the Default model
        self.model_used = self.model_selection.get("ConvLSTM")

        # BACKEND ================= Results dataframe will be returned from the prediction model output
        self.result_data = []
        self.input_video_path = "" #ADDED BY JIAWEN
        ### --------------------------------- ###

        self.video_inputVideo.setStyleSheet("background-color: black")
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(self.video_inputVideo)          # The Video
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)   # If the video is playing or paused
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        # self.video_inputVideo.error.connect(self.handleError)

        self.slider_duration.setRange(0, 0)
        self.slider_duration.sliderMoved.connect(self.setPosition)


        # Result Table
        # Resize the headers: https://stackoverflow.com/questions/38098763/pyside-pyqt-how-to-make-set-qtablewidget-column-width-as-proportion-of-the-a
        table_row_header = self.table_result.verticalHeader()
        table_column_header =  self.table_result.horizontalHeader()
        
        table_row_header.setDefaultSectionSize(250)
        table_column_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        
        ##---- Handle Button Clicks Start---##

        # Load Video using File Explorer
        self.btn_loadVideo.clicked.connect(self.clickOpenFile)

        # Play / Pause Button
        btnSize = QSize(16, 16)
        self.btn_play.setEnabled(False)
        self.btn_play.setFixedHeight(24)
        self.btn_play.setIconSize(btnSize)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setText("Play")
        self.btn_play.clicked.connect(self.playVideo)


        # Run Prediction Button
        self.btn_run.clicked.connect(self.predict)

        # Return Results
        self.btn_results.clicked.connect(lambda: self.show_results(self.result_data))

        # Reset and predict another video
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self.reset_prediction)

        ##---- Handle Button Clicks End ---##
        self.show() # Show the GUI

        
    ## --- Button On Click Functions --- ##

    # Load Video File onto the filepath text and also the Media Player
    def clickOpenFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        print(filename)
        if filename != '':
            self.input_video_path = filename
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename)))
            self.btn_play.setEnabled(True)
            self.lineEdit_path.setText(filename)
            self.lineEdit_path.setEnabled(False)
            self.playVideo()

            # Make Model Selection Visible
            self.radio_Bayesian.setEnabled(True)
            self.radio_LSTM.setEnabled(True)
            self.radio_ConvLSTM.setEnabled(True)

            # LSTM is the default model
            self.radio_ConvLSTM.setChecked(True)
            self.model_used = self.model_selection.get("ConvLSTM")

            self.btn_run.setEnabled(True)

    # Perform Backend Prediction and Output
    def predict(self):
        # Make Model Selection Disabled
        self.radio_Bayesian.setEnabled(False)
        self.radio_LSTM.setEnabled(False)
        self.radio_ConvLSTM.setEnabled(False)

        start = time.time()

        if self.radio_Bayesian.isChecked():
            self.model_used = self.model_selection.get("Bayes")
        elif self.radio_LSTM.isChecked():
            self.model_used = self.model_selection.get("LSTM")
        elif self.radio_ConvLSTM.isChecked():
            self.model_used = self.model_selection.get("ConvLSTM")
            output_dir = os.path.join(PREDICTION_DIR, 'output')
            self.result_data = run_prediction(self.input_video_path, True, output_dir, True)

        ############# Backend: Perform Model Prediction Here ##################
        self.progressBar.setEnabled(True)
        print("Model Used: %s"%(self.model_used))

        # 1. Load the model based on the radio button

        # 2. Load the Video from path to feed the model input

        # 3. Update the progress bar as the prediction is running
        self.completed = 0
        while self.completed <= 100:
            self.completed += 0.001
            self.progressBar.setValue(self.completed)
        print("Completed Prediction")

        # 4. When progress bar is 100%, enable show result
        if self.completed >= 100:
            self.btn_run.setEnabled(False)
            self.btn_results.setEnabled(True)

        # 5. In main UI, Clicking On Show Results will popup a modal of table of ids and result
        self.btn_results.setEnabled(True)

        # 6. Load the output dataframe to the current UI
        # self.result_data = [
        #     {"ID":"Person 1",
        #     "filepath":"sadcat.jpg",
        #     "mood":"Sad",
        #     "conf":"95%"},

        #     {"ID":"Person 2",
        #     "filepath":"sadcat.jpg",
        #     "mood":"Happy",
        #     "conf":"40%"},
        # ]
        end = time.time()
        elapsed = end - start
        self.text_elapsed.setText("Elapsed Time: %s seconds"%(elapsed))
        self.btn_reset.setEnabled(False)

        #######################################################################

    # Populate the result table
    def show_results(self, dataframe):
        print("Dataframe:", dataframe)

        for i in range(len(dataframe)):
            # Create a new row
            row = self.table_result.rowCount()

            # Insert New Row
            self.table_result.insertRow(row)

            # In that new row, insert columns
            self.table_result.setItem(row , 0, QTableWidgetItem(dataframe[i]["ID"]))
            self.table_result.setCellWidget(row , 1, ImgWidget(dataframe[i]["filepath"], self))
            self.table_result.setItem(row , 2, QTableWidgetItem(dataframe[i]["mood"]))
            self.table_result.setItem(row , 3, QTableWidgetItem(dataframe[i]["conf"]))
        
        self.btn_results.setEnabled(False)
        self.btn_reset.setEnabled(True)
        
    # Reset everything for new prediction
    def reset_prediction(self):
        self.result_data = []

        # Make Model Selection DisableD
        self.radio_Bayesian.setEnabled(False)
        self.radio_LSTM.setEnabled(False)
        self.radio_ConvLSTM.setEnabled(False)

        # Reset the buttons and fields
        self.btn_run.setEnabled(False)
        self.completed = 0
        self.progressBar.setValue(self.completed)
        self.btn_results.setEnabled(False)

        self.lineEdit_path.setText("")
        self.lineEdit_path.setEnabled(True)

        self.mediaPlayer.stop()

        self.text_elapsed.setText("Elapsed Time:")

        # Reset the Table
        while (self.table_result.rowCount() > 0):
            self.table_result.removeRow(0)

        self.btn_reset.setEnabled(False)

    # Play/Pause the video when button play is clicked
    def playVideo(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play() 

    # Change the icon of button when play/pause button is clicked
    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.btn_play.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
            self.btn_play.setText("Pause")
        else:
            self.btn_play.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
            self.btn_play.setText("Play")

    # Reset the slider pointer location
    def positionChanged(self, position):
        self.slider_duration.setValue(position)
    
    # Rest the slider pointer remaining time
    def durationChanged(self, duration):
        self.slider_duration.setRange(0, duration)

    # Reset the video to the correct duration point
    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)


        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # MainWindow = uic.loadUi("0_mainUI.ui")
    MainWindow = MainUI()
    sys.exit(app.exec_())