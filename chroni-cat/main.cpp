#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

class AbstractCatDetector
{
protected:
    std::string catImagesDirectory;
    size_t batchSize;
    public:
    size_t readCatImageCount;
    protected:
    size_t detectorPassedCatImageCount;
    size_t imageWithCatDetectedCount;
    std::vector<std::string> catImagePaths;
    std::vector<int> catDetectionMilisecondTimes;
public:
        AbstractCatDetector(std::string catImagesDirectory, int batchSize)
        : catImagesDirectory(catImagesDirectory), batchSize(batchSize), readCatImageCount(0), detectorPassedCatImageCount(0), imageWithCatDetectedCount(0)
        {
            for(const auto& file: std::filesystem::directory_iterator(catImagesDirectory))
            {
                if(".jpg" == file.path().extension())
                {
                    // std::cout << file.path();
                    catImagePaths.push_back(file.path());
                }
            }
        }

        size_t getImageWithCatDetectedCount() { return imageWithCatDetectedCount; }
protected:
        virtual void readCatBatch() = 0;
        virtual size_t detectCatsInBatch() = 0;
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
                imageWithCatDetectedCount += detectCatsInBatch();
                const auto t2 = high_resolution_clock::now();
                const auto time_delta = t2 - t1;
                catDetectionMilisecondTimes.push_back(
                    duration_cast<milliseconds>(time_delta).count()
                    );
            }
        }
};

class OpenCVCatDetector : public AbstractCatDetector
{
public:
    CascadeClassifier classifier;
    std::vector<Mat> catImageBatch;
    OpenCVCatDetector(std::string catImagesDirectory, int batchSize, CascadeClassifier classifier) // move semantics still possible?
    : AbstractCatDetector(catImagesDirectory, batchSize), classifier(classifier)
    { }
protected: // private?
    virtual void readCatBatch()
    {
        catImageBatch.clear();
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
                catImageBatch.push_back(catImage);
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
    virtual size_t detectCatsInBatch()
    {
        size_t detectedCats = 0;
        for(size_t i = 0;
            i < batchSize && detectorPassedCatImageCount < catImagePaths.size();
            i++, detectorPassedCatImageCount++
        )
        {
            detectedCats += detectCat(catImageBatch[i]);
        }
        return detectedCats;
    }
};

int main(int argc, const char** argv)
{
    const size_t batchSize = 64;
    String catFaceCascadePath = samples::findFile("data/haarcascades/haarcascade_frontalcatface.xml");
    CascadeClassifier catFaceCascadeClassifier;
    catFaceCascadeClassifier.load(catFaceCascadePath);
    OpenCVCatDetector CVDetector = OpenCVCatDetector("Cats_Dataset/Images", batchSize, catFaceCascadeClassifier);
    CVDetector.detectCatsInCatImages();
    std::cout << CVDetector.readCatImageCount << std::endl;
    std::cout << CVDetector.getImageWithCatDetectedCount() << std::endl;
    return 0;
}
