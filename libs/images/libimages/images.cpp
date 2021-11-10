#include "images.h"

#include <cassert>

// Configuring display system in CImg (See http://cimg.eu/reference/group__cimg__environment.html):
#ifdef __unix__
    // X11-based
    #define cimg_display 1
#elif defined _WIN32
    #define cimg_display 2
#else
    #define cimg_display 0
#endif
// CImg header-only library available at https://github.com/dtschump/CImg (v.2.3.4 0f0d65d984b08ad8178969f4fa4d1641d721354b)
#include "CImg.h"
#undef cimg_display

using namespace images;
using namespace cimg_library;

template <typename T>
Image<T>::Image()
        : width(0), height(0), cn(0), offset(0), stride(0), data(NULL) {}

template <typename T>
Image<T>::Image(size_t width, size_t height, size_t cn, const std::shared_ptr<T> &data) : Image(width, height, cn, data, 0, width * cn) {}

template <typename T>
Image<T>::Image(size_t width, size_t height, size_t cn, const std::shared_ptr<T> &data, size_t offset, ptrdiff_t stride)
        : width(width), height(height), cn(cn), offset(offset), stride(stride), data(data) {
    if (!this->data) {
        allocateData();
    }
}

template <typename T>
void Image<T>::allocateData() {
    this->data = std::shared_ptr<T>(new T[width * height * cn]);
}

template <typename T>
Image<T>::Image(const Image<T> &that) : width(that.width), height(that.height), cn(that.cn), offset(that.offset), stride(that.stride), data(that.data) {}

template <typename T>
Image<T>::Image(const char *const filename) {
    try {
        CImg<T> img(filename);
        CImgWrapper<T> wrapper(img);
        fromCImg(wrapper);
    } catch (CImgIOException& e) {
        width = 0;
        height = 0;
        cn = 0;
        offset = 0;
        stride = 0;
        data = NULL;
        return;
    }
}

template <typename T>
Image<T>::Image(const std::string& filename) {
    try {
        CImg<T> img(filename.c_str());
        CImgWrapper<T> wrapper(img);
        fromCImg(wrapper);
    } catch (CImgIOException& e) {
        width = 0;
        height = 0;
        cn = 0;
        offset = 0;
        stride = 0;
        data = NULL;
        return;
    }
}

template <typename T>
void Image<T>::fromCImg(CImgWrapper<T>& wrapper) {
    CImg<T>& img = *wrapper.img;

    width = img.width();
    height = img.height();
    cn = img.spectrum();
    offset = 0;
    stride = width * cn;
    allocateData();

    T* src = img.data();
    for (size_t c = 0; c < cn; c++) {
        T* dst = this->ptr() + c;
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                *dst = *src;
                ++src;
                dst += cn;
            }
        }
    }
}

template <typename T>
CImgWrapper<T> Image<T>::toCImg() {
    CImgWrapper<T> wrapper(width, height, cn);
    CImg<T>& img = *wrapper.img;

    T* dst = img.data();
    for (size_t c = 0; c < cn; c++) {
        T* src = ptr() + c;
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                *dst = *src;
                ++dst;
                src += cn;
            }
        }
    }

    return wrapper;
}

template <typename T>
Image<T> Image<T>::copy() const {
    Image<T> result(width, height, cn);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            for (size_t c = 0; c < cn; c++) {
               result(y, x, c) = this->operator()(y, x, c);
            }
        }
    }
    return result;
}

template<typename T>
Image<T> Image<T>::reshape(size_t width, size_t height, size_t cn) {
    assert (this->width * this->height * this->cn == width * height * cn);
    return Image<T>(width, height, cn, this->data, this->offset, this->stride);
}

template<typename T>
Image<T> Image<T>::getCrop(size_t offsetRow, size_t offsetCol, size_t height, size_t width) {
    assert (offsetRow + height <= this->height && offsetCol + width <= this->width);

    Image<T> part(width, height, this->cn);
    for (size_t y = offsetRow; y < offsetRow + height; y++) {
        for (size_t x = offsetCol; x < offsetCol + width; x++) {
            for (size_t c = 0; c < this->cn; c++) {
               part(y - offsetRow, x - offsetCol, c) = this->operator()(y, x, c);
            }
        }
    }
    return part;
}

template<typename T>
Image<T> Image<T>::removeAlphaChannel() {
    assert(cn == 4);

    Image<T> result(width, height, 3);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            for (size_t c = 0; c < 3; c++) {
               result(y, x, c) = this->operator()(y, x, c);
            }
        }
    }
    return result;
}

template <typename T>
Image<T> Image<T>::resize(size_t width, size_t height) {
    CImgWrapper<T> wrapper = this->toCImg();
    CImg<T>& img = *wrapper.img;

    int pdx = width;
    int pdy = (height == 0) ? -100 : height;
    int pdz = -100;
    int pdv = -100;
    unsigned int interp = (width > this->width) ? 1 : 3;
    img = img.resize(pdx, pdy, pdz, pdv, interp);

    Image<T> resized = Image<T>();
    resized.fromCImg(wrapper);
    return resized;
}

template <typename T>
void Image<T>::fill(T value) {
    assert (cn == 1);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            for (size_t c = 0; c < cn; c++) {
                this->operator()(y, x, c) = value;
            }
        }
    }
}

template <typename T>
void Image<T>::fill(T value[]) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            for (size_t c = 0; c < cn; c++) {
                this->operator()(y, x, c) = value[c];
            }
        }
    }
}

template <typename T>
void Image<T>::replace(T a, T b) {
    assert (cn == 1);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (this->operator()(y, x) == a) {
                this->operator()(y, x) = b;
            }
        }
    }
}

template <typename T>
void Image<T>::replace(T a[], T b[]) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            bool match = true;
            for (size_t c = 0; c < cn; c++) {
                if (this->operator()(y, x, c) != a[c]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                for (size_t c = 0; c < cn; c++) {
                    this->operator()(y, x, c) = b[c];
                }
            }
        }
    }
}

template <typename T>
ImageWindow Image<T>::show(const char *title) const {
    ImageWindow window = ImageWindow(title);
    window.display(*this);
    return window;
}

template <typename T>
void Image<T>::saveJPEG(const char *const filename, int quality) {
    CImgWrapper<T> wrapper = this->toCImg();
    wrapper.img->save_jpeg(filename, quality);
}

template <typename T>
void Image<T>::saveJPEG(const std::string& filename, int quality) {
    saveJPEG(filename.c_str(), quality);
}

template <typename T>
void Image<T>::savePNG(const char *const filename) {
    CImgWrapper<T> wrapper = this->toCImg();
    wrapper.img->save_png(filename);
}

template <typename T>
void Image<T>::savePNG(const std::string& filename) {
    savePNG(filename.c_str());
}

class CImgDisplayWrapper {
public:
    CImgDisplay display;
};

ImageWindow::ImageWindow(std::string title) : title(title) {
    cimg_display = new CImgDisplayWrapper();
    setTitle(title);
}

ImageWindow::~ImageWindow() {
    delete cimg_display;
}

template<typename T>
void ImageWindow::display(Image<T> image) {
    CImgWrapper<T> wrapper = image.toCImg();
    cimg_display->display.display(*wrapper.img);
    setTitle(title);
}

void ImageWindow::resize(size_t width, size_t height) {
    cimg_display->display.resize(width, height);
}

void ImageWindow::resize() {
    cimg_display->display.resize();
}

void ImageWindow::wait(unsigned int milliseconds) {
    cimg_display->display.wait(milliseconds);
}

void ImageWindow::setTitle(std::string title) {
    cimg_display->display.set_title(title.data());
}

bool ImageWindow::isClosed() {
    return cimg_display->display.is_closed();
}

bool ImageWindow::isResized() {
    return cimg_display->display.is_resized();
}

mouse_click_t ImageWindow::getMouseClick() {
    return cimg_display->display.button();
}

int ImageWindow::getMouseX() {
    return cimg_display->display.mouse_x();
}

int ImageWindow::getMouseY() {
    return cimg_display->display.mouse_y();
}

size_t ImageWindow::width() {
    return cimg_display->display.window_width();
}

size_t ImageWindow::height() {
    return cimg_display->display.window_height();
}




template void ImageWindow::display<unsigned char>(Image<unsigned char> image);
template void ImageWindow::display<unsigned short>(Image<unsigned short> image);
template void ImageWindow::display<float>(Image<float> image);
template class images::Image<unsigned char>;
template class images::Image<unsigned short>;
template class images::Image<float>;
