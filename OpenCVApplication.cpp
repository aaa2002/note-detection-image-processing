#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 
#include <windows.h>

using namespace std;

wchar_t* projectPath;

#define C4 261.63
#define D4 293.66
#define E4 329.63
#define F4 349.23
#define G4 392.00
#define A4 440.00
#define B4 493.88
#define C5 523.25
#define D5 587.33
#define E5 659.25
#define F5 698.46
#define G5 783.99

struct Note {
    string duration;
    int pitch;
    int x;
    int y;
};

Mat_<uchar> wholeNoteTemplate = imread("Sample Images/templates/whole.png", IMREAD_GRAYSCALE);
Mat_<uchar> halfNoteTemplate = imread("Sample Images/templates/half.png", IMREAD_GRAYSCALE);
Mat_<uchar> halfNoteLinedTemplate = imread("Sample Images/templates/half2.png", IMREAD_GRAYSCALE);
Mat_<uchar> quarterNoteTemplate = imread("Sample Images/templates/quarter.png", IMREAD_GRAYSCALE);
Mat_<uchar> quarterNoteBreakTemplate = imread("Sample Images/templates/qbreak.png", IMREAD_GRAYSCALE);

void playBeep(float pitch, string duration) {
    int durationMs = 0;

    if (duration == "Whole") {
		durationMs = 2000;
	}
    else if (duration == "Half") {
		durationMs = 1000;
	}
    else if (duration == "Quarter") {
		durationMs = 500;
	}

    Beep(pitch, durationMs);
}

Mat_<uchar> histogram(Mat_<uchar> img)
{
    Mat_<uchar> hist(img.rows, img.cols, uchar(255));

    for (int i = 0; i < img.rows; i++)
    {
        vector<uchar> crtRow;
        for (int j = 0; j < img.cols; j++)
        {
            if (img(i, j) == 0)
            {
                crtRow.push_back(0);
            }
        }

        if (crtRow.size() < img.cols)
        {
            int diff = img.cols - crtRow.size();
            for (int k = 0; k < diff; k++)
            {
                crtRow.push_back(255);
            }
        }

        for (int j = 0; j < img.cols; j++)
        {
            hist(i, j) = crtRow[j];
        }
    }

    return hist;
}

Mat_<uchar> adaptiveThresholdMean(const Mat_<uchar>& img, int block_size, double c) {
    Mat_<uchar> binary(img.size());

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int x_min = max(0, i - block_size / 2);
            int y_min = max(0, j - block_size / 2);
            int x_max = min(img.rows - 1, i + block_size / 2);
            int y_max = min(img.cols - 1, j + block_size / 2);
            Mat_<uchar> block = img(Range(x_min, x_max + 1), Range(y_min, y_max + 1));
            double thresh = mean(block)[0] - c;
            if (img(i, j) >= thresh) {
                binary(i, j) = 255;
            }
            else {
                binary(i, j) = 0;
            }
        }
    }

    return binary;
}

Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.rows, src.cols);
    dst.setTo(255);

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (src(i, j) == 0)
            {
                for (int u = 0; u < strel.rows; u++)
                {
                    for (int v = 0; v < strel.cols; v++)
                    {
                        if (strel(u, v) == 0)
                        {
                            int i2 = i + u - strel.rows / 2;
                            int j2 = j + v - strel.cols / 2;

                            if (i2 >= 0 && i2 < src.rows && j2 >= 0 && j2 < src.cols)
                            {
                                dst(i2, j2) = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    return dst;
}

Mat_<uchar> erosion(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.rows, src.cols);
    dst.setTo(255);

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (src(i, j) == 0)
            {
                bool allBlack = true;
                for (int u = 0; u < strel.rows; u++)
                {
                    for (int v = 0; v < strel.cols; v++)
                    {
                        if (strel(u, v) == 0)
                        {
                            int i2 = i + u - strel.rows / 2;
                            int j2 = j + v - strel.cols / 2;

                            if (i2 >= 0 && i2 < src.rows && j2 >= 0 && j2 < src.cols)
                            {
                                if (src(i2, j2) == 255)
                                {
                                    allBlack = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (!allBlack)
                        break;
                }

                if (allBlack)
                {
                    dst(i, j) = 0;
                }
            }
        }
    }

    return dst;
}

float calculateBlackDensity(const Mat_<uchar>& src, int x_min, int y_min, int x_max, int y_max) {
    int blackCount = 0;
    int totalPixels = (x_max - x_min + 1) * (y_max - y_min + 1);

    for (int i = y_min; i <= y_max; ++i) {
        for (int j = x_min; j <= x_max; ++j) {
            if (src(i, j) == 0) {
                blackCount++;
            }
        }
    }

    return (float) blackCount / (float) totalPixels;
}

vector<int> selectOneLine(vector<int>& input) {
    // helper function (needed because each line is represented as 2 actual lines, in order to make them easier to see in the output image)
    if (input.empty()) return {};
    unique(input.begin(), input.end());
    vector<int> result;
    result.push_back(input[0]);

    for (size_t i = 1; i < input.size(); ++i) {
        if (input[i] != input[i - 1] + 1 && (result.empty() || input[i] != result.back() + 1)) {
            result.push_back(input[i]);
        }
    }

    return result;
}

vector<int> detectStaffLines(const Mat_<uchar>& img, Mat_<Vec3b>& output) {
    Mat_<uchar> hist = histogram(img);
    vector<int> histogramValues;

    for (int i = 0; i < hist.rows; i++) {
        int count = 0;
        for (int j = 0; j < hist.cols; j++) {
            if (hist(i, j) == 0) {
                count++;
            }
        }
        histogramValues.push_back(count);
    }

    int maxVal = *max_element(histogramValues.begin(), histogramValues.end());
    vector<int> staffLinesLocationsVertical;

    for (int i = 0; i < histogramValues.size(); i++) {
        if (histogramValues[i] > 0.6 * maxVal) {
            staffLinesLocationsVertical.push_back(i);
        }
    }

    for (int i = 0; i < staffLinesLocationsVertical.size(); i++) {
        for (int j = 0; j < img.cols; j++) {
            output(staffLinesLocationsVertical[i], j) = Vec3b(0, 255, 0);
        }
    }

    vector<int> newStaffLines;
    staffLinesLocationsVertical = selectOneLine(staffLinesLocationsVertical);
    for (int i = 0; i < staffLinesLocationsVertical.size() - 1; i++) {
        newStaffLines.push_back(staffLinesLocationsVertical[i]);
        newStaffLines.push_back((staffLinesLocationsVertical[i] + staffLinesLocationsVertical[i + 1]) / 2);
    }
    newStaffLines.push_back(staffLinesLocationsVertical.back());
    newStaffLines.push_back(staffLinesLocationsVertical.back() + (newStaffLines[1] - newStaffLines[0]));
    return newStaffLines;
}

int determinePitch(int y, const vector<int>& staffLines) {
    vector<float> pitches = { G5, F5, E5, D5, C5, B4, A4, G4, F4, E4, D4 };
    for (int i = 0; i < staffLines.size(); ++i) {
        if (y < staffLines[i]) {
            cout << "Line: " << staffLines[i] << " Y: " << y << '\n';
            return (int)pitches[i];
        }
    }
    return pitches.back();
}

bool isInside(int i, int j, int rows, int cols) {
    return i >= 0 && j >= 0 && i < rows && j < cols;
}

vector<Note> manualTemplateMatching(const Mat_<uchar>& img, const vector<int>& staffLines, const Mat_<uchar>& templateImg, const string& noteType, Mat_<Vec3b>& output, double threshold = 0.85) {
    vector<Note> notes;
    vector<Rect> boxes;
    vector<float> scores;

    for (int i = 0; i < img.rows; ++i) {

        for (int j = 0; j < img.cols; ++j) {
            int matchCount = 0;

            for (int y = 0; y < templateImg.rows; ++y) {
                for (int x = 0; x < templateImg.cols; ++x) {
                    int imgY = i + y;
                    int imgX = j + x;
                    if (isInside(imgY, imgX, img.rows, img.cols) && img(imgY, imgX) == templateImg(y, x)) {
                        matchCount++;
                    }
                }
            }

            float matchPercentage = (float)(matchCount) / (templateImg.rows * templateImg.cols);
            if (matchPercentage >= threshold) {
                Rect boundingBox(j, i, templateImg.cols, templateImg.rows);
                if (boundingBox.x + boundingBox.width <= output.cols && boundingBox.y + boundingBox.height <= output.rows) {
                    boxes.push_back(boundingBox);
                    scores.push_back(matchPercentage);
                }
            }
        }
    }
    
    vector<int> indices;
    dnn::NMSBoxes(boxes, scores, threshold, 0.2, indices);

    for (int idx : indices) {
        Rect box = boxes[idx];

        int pitch = determinePitch(box.y, staffLines);
        notes.push_back({ noteType, pitch, box.x, box.y });

        rectangle(output, box, Scalar(0, 255, 0), 2);
        putText(output, noteType, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 50, 0), 1);
    }
    
    imshow("img", output);
    cout << "Detection for " << noteType << " notes done.\n";
    return notes;
}

//initial implementation used simpleBlobDetection function to detect notes, but that proved to be inaccurate
/*void simpleBlobDetection(const Mat_<uchar>& img, Mat_<Vec3b>& output) {
    cvtColor(img, output, COLOR_GRAY2BGR);
    Mat_<int> labels(img.size(), 0);
    int currentLabel = 1;

    int dx[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
    int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0 && labels(i, j) == 0) {
                queue<Point> q;
                q.push(Point(j, i));
                labels(i, j) = currentLabel;

                int minX = j, minY = i, maxX = j, maxY = i;

                while (!q.empty()) {
                    Point p = q.front();
                    q.pop();

                    for (int k = 0; k < 8; k++) {
                        int nx = p.x + dx[k];
                        int ny = p.y + dy[k];
                        if (nx >= 0 && nx < img.cols && ny >= 0 && ny < img.rows && img(ny, nx) == 0 && labels(ny, nx) == 0) {
                            q.push(Point(nx, ny));
                            labels(ny, nx) = currentLabel;
                            minX = min(minX, nx);
                            minY = min(minY, ny);
                            maxX = max(maxX, nx);
                            maxY = max(maxY, ny);
                        }
                    }
                }

                float area = (maxX - minX + 1) * (maxY - minY + 1);
                float blackDensity = calculateBlackDensity(img, minX, minY, maxX, maxY);
                float aspectRatio = static_cast<float>(maxX - minX + 1) / (maxY - minY + 1);
                int centerX = (minX + maxX) / 2;
                int centerY = (minY + maxY) / 2;

                string noteType;
                if (blackDensity < 0.7) {
                    noteType = "Half";
                }
                else {
                    if (img(centerY, centerX) == 255) {
                        noteType = "Whole";
                    }
                    else {
                        noteType = "Quarter";
                    }
                }


                rectangle(output, Point(minX, minY), Point(maxX, maxY), Scalar(0, 0, 255), 1);
                putText(output, noteType, Point(minX, maxY + 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);

                currentLabel++;
            }
        }
    }

}*/

Mat_<uchar> filterByCircularity(const Mat_<uchar>& input) {
    Mat_<float> circularity(input.size(), 0.0);
    Mat_<uchar> output(input.size(), 255); 

    for (int y = 1; y < input.rows - 1; ++y) {
        for (int x = 1; x < input.cols - 1; ++x) {
            float Ixx = (input(y, x + 1) + input(y, x - 1) - 2 * input(y, x)) / 2.0;
            float Iyy = (input(y + 1, x) + input(y - 1, x) - 2 * input(y, x)) / 2.0;
            float Ixy = (input(y + 1, x + 1) + input(y - 1, x - 1) - input(y - 1, x + 1) - input(y + 1, x - 1)) / 4.0;

            float denominator = sqrt(pow((Ixx - Iyy), 2) + 4 * pow(Ixy, 2));
            if (denominator != 0.0)
                circularity(y, x) = (Ixx + Iyy + denominator) / (Ixx + Iyy - denominator);
        }
    }

    float threshold = 0.2;
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            if (circularity(y, x) < threshold) {
                output(y, x) = input(y, x);
            }
        }
    }

    return output;
}

void mergeCloseNotes(vector<Note>& notes, int threshold) {
    // helper function (needed because when template matching, some notes are detected multiple times - they match consecutive positions)
    vector<Note> mergedNotes;
    sort(notes.begin(), notes.end(), [](const Note& a, const Note& b) {
        return a.x < b.x;
        });

    for (int i = 0; i < notes.size(); ++i) {
        bool merged = false;
        for (int j = 0; j < mergedNotes.size(); ++j) {
            if (abs(notes[i].x - mergedNotes[j].x) < threshold) {
                mergedNotes[j].duration = notes[i].duration;
                merged = true;
                break;
            }
        }
        if (!merged) {
            mergedNotes.push_back(notes[i]);
        }
    }

    notes = mergedNotes;
}

void testOpenImage()
{
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat_<uchar> src;
        src = imread(fname, IMREAD_GRAYSCALE);
        //imshow("src", src);
        Mat_<uchar> hist = histogram(src);
        
        Mat_<uchar> dst;
        // manualAdaptiveThreshold(~src, dst, 3, 255, 10);
        // adaptiveThreshold(~src, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
        //dst = adaptiveThresholdMean(src, 15, -2);
        dst = src.clone();
        // imshow("image", dst);


        Mat_<uchar> horizontal = dst.clone();
        Mat_<uchar> vertical = dst.clone();
        int horizontalsize = horizontal.cols / 30;
        int verticalsize = vertical.rows / 30;  

        Mat_<uchar> verticalStructure(3, 3, (uchar)0);
        Mat_<uchar> horizontalStructure(1, 3, (uchar)0);
        Mat_<uchar> verticalStructure2(3, 1, (uchar)0);
        Mat_<uchar> horizontalStructure2(1, 3, (uchar)0);

        Mat_<uchar> erodedHorizontal = erosion(dst, horizontalStructure);
        // imshow("ErodedH", erodedHorizontal);
        Mat_<uchar> dilatedHorizontal = dilation(erodedHorizontal, horizontalStructure);
        // imshow("DilatedH", dilatedHorizontal);

        
        Mat_<uchar> erodedVertical = erosion(dilatedHorizontal, verticalStructure);
        // imshow("ErodedV", erodedVertical);
        Mat_<uchar> dilatedVertical = dilation(erodedVertical, verticalStructure);
        //imshow("DilatedV", dilatedVertical); 
        //imwrite("filename.png", dilatedVertical);

        Mat_<uchar> filtered = filterByCircularity(dilatedVertical);

        Mat_<Vec3b> blobOutput(src.rows, src.cols);
        cvtColor(src, blobOutput, COLOR_GRAY2BGR);

        //simpleBlobDetection(filtered, blobOutput);
        vector<int> staffLines = detectStaffLines(src, blobOutput);
        //imshow("staff", blobOutput);
       
        
        vector<Note> detectedNotes1, detectedNotes2, detectedNotes3, detectedNotes, detectedNotes4;
        detectedNotes1 = manualTemplateMatching(filtered, staffLines, wholeNoteTemplate, "Whole", blobOutput);
        detectedNotes2 = manualTemplateMatching(filtered, staffLines, halfNoteTemplate, "Half", blobOutput);
        detectedNotes4 = manualTemplateMatching(dilatedVertical, staffLines, halfNoteLinedTemplate, "Half", blobOutput);
        detectedNotes3 = manualTemplateMatching(filtered, staffLines, quarterNoteTemplate, "Quarter", blobOutput);
        manualTemplateMatching(filtered, staffLines, quarterNoteBreakTemplate, "Break", blobOutput);
        //imwrite("f.png", filtered);

        detectedNotes.insert(detectedNotes.end(), detectedNotes1.begin(), detectedNotes1.end());
        detectedNotes.insert(detectedNotes.end(), detectedNotes2.begin(), detectedNotes2.end());
        detectedNotes.insert(detectedNotes.end(), detectedNotes3.begin(), detectedNotes3.end());
        detectedNotes.insert(detectedNotes.end(), detectedNotes4.begin(), detectedNotes4.end());

        mergeCloseNotes(detectedNotes, 15);

        cout << "Detected Notes:" << endl;
        cout << "Number of detected notes: " << detectedNotes.size() << endl;
        for (const auto& note : detectedNotes) {
            cout << "Duration: " << note.duration << ", Pitch: " << note.pitch << ", X: " << note.x << ", Y: " << note.y << '\n';
            playBeep(note.pitch, note.duration);
        }

        waitKey();
    }
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    int op;
    do
    {
        system("cls");
        destroyAllWindows();
        printf("Menu:\n");
        printf(" 1 - test\n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);
        switch (op)
        {
        case 1:
            testOpenImage();
            break;
        }
    } while (op != 0);
    return 0;
}
