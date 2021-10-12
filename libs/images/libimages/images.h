#pragma once

#include <memory>
#include <cstddef>
#include <cassert>
#include <string>

namespace cimg_library {
    template<typename T>
    class CImg;
}

using cimg_library::CImg;

template <typename T>
class CImgWrapper {
public:
    std::shared_ptr<CImg<T>> img;

    CImgWrapper(CImg<T>& img) : img(std::make_shared<CImg<T>>(img)) {}
    CImgWrapper(size_t width, size_t height, size_t cn) : img(std::make_shared<CImg<T>>(width, height, 1, cn)) {}
};

typedef int mouse_click_t;

const mouse_click_t MOUSE_LEFT = 1;
const mouse_click_t MOUSE_RIGHT = 2;
const mouse_click_t MOUSE_MIDDLE = 4;

class CImgDisplayWrapper;

namespace images {

    template <typename T>
    class Image;

    class ImageWindow {
    public:
        ImageWindow(std::string title);
        ~ImageWindow();

        template<typename T>
        void display(Image<T> image);
        void resize(size_t width, size_t height);
        void resize();
        void wait(unsigned int milliseconds);
        void setTitle(std::string title);

        bool isClosed();
        bool isResized();

        mouse_click_t getMouseClick();
        int getMouseX();
        int getMouseY();

        size_t width();
        size_t height();

    protected:
        CImgDisplayWrapper* cimg_display;
        std::string title;
    };

    template <typename T>
    class Image {
    public:
        size_t width;
        size_t height;
        size_t cn;

        Image();
        Image(size_t width, size_t height, size_t cn, const std::shared_ptr<T> &data=nullptr);
        Image(size_t width, size_t height, size_t cn, const std::shared_ptr<T> &data, size_t offset, ptrdiff_t stride);
        Image(const Image<T> &image);
        Image(const char *const filename);
        Image(const std::string& filename);

        void fromCImg(CImgWrapper<T>& wrapper);
        CImgWrapper<T> toCImg();

        Image copy() const;
        Image<T>& operator=(const Image<T>& that) = default;
        Image<T> reshape(size_t width, size_t height, size_t cn);
        Image<T> getCrop(size_t offsetRow, size_t offsetCol, size_t height, size_t width);
        Image<T> removeAlphaChannel();

        Image<T> resize(size_t width, size_t height=0);

        void fill(T value);
        void fill(T value[]);
        void replace(T a, T b);
        void replace(T a[], T b[]);

        ImageWindow show(const char* title) const;

        void saveJPEG(const char *const filename, int quality=100);
        void saveJPEG(const std::string& filename, int quality=100);
        void savePNG(const char *const filename);
        void savePNG(const std::string& filename);

        bool isNull()   { return !((bool) data);    };
        T* ptr()        { return data.get();        }

        inline T& operator()(size_t row, size_t col) {
            assert (row < height && col < width);
            return data.get()[offset + row * stride + col * cn];
        }

        inline T& operator()(size_t row, size_t col, size_t c) {
            assert (c >= 0 && c < cn);
            return data.get()[offset + row * stride + col * cn + c];
        }

        inline T operator()(size_t row, size_t col) const {
            assert (row < height && col < width);
            return data.get()[offset + row * stride + col * cn];
        }

        inline T operator()(size_t row, size_t col, size_t c) const {
            assert (c >= 0 && c < cn);
            return data.get()[offset + row * stride + col * cn + c];
        }

    protected:
        size_t offset;
        ptrdiff_t stride;

        std::shared_ptr<T> data;

        void allocateData();
    };

}
