#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

enum ImageClass { ClassCat, ClassNonCat };

class AbstractCatDetector
{
protected:
    // std::string catImagesDirectory;
    size_t batchSize;
    public:
    size_t readCatImageCount;
    size_t readNonCatImageCount;
    protected:
    size_t detectorProcessedImgsCount;
    size_t correctPredictionCount;
    std::vector<std::string> catImagePaths;
    std::vector<std::string> nonCatImagePaths;
    std::vector<int> batchDetectionTimesMs;
public:
        AbstractCatDetector(std::string catImagesDirectory, std::string nonCatImagesDirectory, int batchSize)
        : batchSize(batchSize), readCatImageCount(0), readNonCatImageCount(0), detectorProcessedImgsCount(0), correctPredictionCount(0)
        {
            for(const auto& file: std::filesystem::directory_iterator(catImagesDirectory))
                if(".jpg" == file.path().extension())
                    catImagePaths.push_back(file.path());

            for(const auto& file: std::filesystem::directory_iterator(nonCatImagesDirectory))
                if(".jpg" == file.path().extension())
                    nonCatImagePaths.push_back(file.path());
            
            std::cout << "catImagePaths: " << catImagePaths.size() << "nonCatImagePaths " << nonCatImagePaths.size() << "\n";
            assert(0 != catImagePaths.size());
            assert(0 != nonCatImagePaths.size());
        }

        size_t getCorrectPredictionCount() { return correctPredictionCount; }
protected:
        virtual void readCatBatch() = 0;
        virtual void readNonCatBatch() = 0;
        virtual size_t batchPredict(enum ImageClass c) = 0;
public:
        virtual void detectCatsInCatImages()
        {
            for(size_t i = 0; i < catImagePaths.size(); i += batchSize)
            {
                using std::chrono::high_resolution_clock;
                using std::chrono::duration_cast;
                using std::chrono::duration;
                using std::chrono::milliseconds;
                readCatBatch();
                const auto t1 = high_resolution_clock::now();
                correctPredictionCount += batchPredict(ClassCat);
                const auto t2 = high_resolution_clock::now();
                const auto time_delta = t2 - t1;
                batchDetectionTimesMs.push_back(
                    duration_cast<milliseconds>(time_delta).count()
                    );
            }
            for(size_t i = 0; i < nonCatImagePaths.size(); i += batchSize)
            {
                std::cout << i << "\n";
                using std::chrono::high_resolution_clock;
                using std::chrono::duration_cast;
                using std::chrono::duration;
                using std::chrono::milliseconds;
                readNonCatBatch();
                const auto t1 = high_resolution_clock::now();
                correctPredictionCount += batchPredict(ClassNonCat);
                const auto t2 = high_resolution_clock::now();
                const auto time_delta = t2 - t1;
                batchDetectionTimesMs.push_back(
                    duration_cast<milliseconds>(time_delta).count()
                    );
            }
        }
        void dumpDetectionTimes(std::string filename)
        {
            std::ofstream f(filename, std::ofstream::out);
            for(ssize_t i = 0; i < batchDetectionTimesMs.size() - 1; i++)
                f << batchDetectionTimesMs[i] << ";";
            f << batchDetectionTimesMs.back() << "\n";
        }
};

class OpenCVCatDetector : public AbstractCatDetector
{
public:
    CascadeClassifier classifier;
    std::vector<Mat> imgBatch;
    OpenCVCatDetector(std::string catImagesDirectory, std::string nonCatImagesDirectory, int batchSize, CascadeClassifier classifier) // move semantics still possible?
    : AbstractCatDetector(catImagesDirectory, nonCatImagesDirectory, batchSize), classifier(classifier)
    { }
protected: // private?
    virtual void readCatBatch()
    {
        imgBatch.clear();
        for (size_t i = 0;
            i < batchSize && readCatImageCount < catImagePaths.size();
            
        )
        {
            // std::cout << catImagePaths[readCatImageCount] << std::endl;
            Mat catImage = imread(catImagePaths[readCatImageCount]);
            if(catImage.empty())
            {
                std::cerr << "Warning: Failed to load " << catImagePaths[readCatImageCount] << "\n";
                catImagePaths.erase(catImagePaths.begin() + readCatImageCount);
            }
            else
            {
                readCatImageCount++;
                i++;
                imgBatch.push_back(catImage);
            }
        }
    }
    virtual void readNonCatBatch()
    {
        imgBatch.clear();
        for (size_t i = 0;
            i < batchSize && readNonCatImageCount < nonCatImagePaths.size();
            
        )
        {
            Mat catImage = imread(nonCatImagePaths[readNonCatImageCount]);
            if(catImage.empty())
            {
                std::cerr << "Warning: Failed to load " << nonCatImagePaths[readNonCatImageCount] << "\n";
                nonCatImagePaths.erase(nonCatImagePaths.begin() + readNonCatImageCount);
            }
            else
            {
                readNonCatImageCount++;
                i++;
                imgBatch.push_back(catImage);
            }
        }
    }
        
    bool detectCat(const Mat& image)
    {
        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        equalizeHist(grayImage, grayImage);
        std::vector<Rect> catFaces;
        classifier.detectMultiScale(grayImage, catFaces);
        // std::cout << catFaces.size() << std::endl << std::endl;
        return catFaces.size();
    }
    virtual size_t batchPredict(enum ImageClass c)
    {
        size_t correctPredicions = 0;
        // for(size_t i = 0;
        //     i < batchSize && detectorProcessedImgsCount < catImagePaths.size();
        //     i++, detectorProcessedImgsCount++
        // )
        for(const auto& img: imgBatch)
        {
            if(ImageClass::ClassCat == c)
                correctPredicions += detectCat(img);
            if(ImageClass::ClassNonCat == c)
                correctPredicions += !detectCat(img);
        }
        // for(size_t i = 0;
        //     i < batchSize && detectorProcessedImgsCount < catImagePaths.size();
        //     i++, detectorProcessedImgsCount++
        // )
        // {
        //     bool correctPrediction = !detectCat(imgBatch[i]);
        //     correctPredicions += correctPrediction;
        // }
        return correctPredicions;
    }
};

int main(int argc, const char** argv)
{
    const size_t batchSize = 128;
    {
        String catFaceCascadePath = samples::findFile("data/haarcascades/haarcascade_frontalcatface.xml");
        CascadeClassifier catFaceCascadeClassifier;
        catFaceCascadeClassifier.load(catFaceCascadePath);
        OpenCVCatDetector CVDetector = OpenCVCatDetector("../cat-fetcher/Cats_Dataset/Images", "../cat-fetcher/niekotki", batchSize, catFaceCascadeClassifier);
        CVDetector.detectCatsInCatImages();

        std::cout << "haarcascade_frontalcatface all cat images: "<< CVDetector.readCatImageCount << std::endl;
        std::cout << "haarcascade_frontalcatface correct predicted images: " << CVDetector.getCorrectPredictionCount() << std::endl;
        CVDetector.dumpDetectionTimes("haarcascade_frontalcatface_times");
    }
    {
        String catFaceCascadePath = samples::findFile("data/haarcascades/haarcascade_frontalcatface_extended.xml");
        CascadeClassifier catFaceCascadeClassifier;
        catFaceCascadeClassifier.load(catFaceCascadePath);
        OpenCVCatDetector CVDetector = OpenCVCatDetector("../cat-fetcher/Cats_Dataset/Images", "../cat-fetcher/niekotki", batchSize, catFaceCascadeClassifier);
        CVDetector.detectCatsInCatImages();

        std::cout << "haarcascade_frontalcatface_extended all cat images: " << CVDetector.readCatImageCount << std::endl;
        std::cout << "haarcascade_frontalcatface_extended correct predicted images: " << CVDetector.getCorrectPredictionCount() << std::endl;
        CVDetector.dumpDetectionTimes("haarcascade_frontalcatface_extended_times");
    }
    return 0;
}
