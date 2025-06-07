#include "task_loadimg.hh"
#include "task_wavelet.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>

using namespace focusstack;

Task_LoadImg::Task_LoadImg(std::string filename, float wait_images)
{
  m_filename = filename;
  m_name = "Load " + filename;
  m_wait_images = wait_images;
  m_wait_images_until = std::chrono::system_clock::now()
                      + std::chrono::milliseconds((int)(m_wait_images * 1000));
}

Task_LoadImg::Task_LoadImg(std::string name, const cv::Mat &img)
{
  m_filename = name;
  m_name = "Memory image " + name;
  m_result = img.clone();
  m_wait_images = 0;
  m_wait_images_until = std::chrono::system_clock::now()
                      + std::chrono::milliseconds((int)(m_wait_images * 1000));
}

bool Task_LoadImg::ready_to_run()
{
  if (!ImgTask::ready_to_run())
  {
    return false;
  }

  // Wait for image files to appear.
  // This is useful for processing images as soon as they appear.
  if (m_wait_images > 0 && std::chrono::system_clock::now() < m_wait_images_until)
  {
    std::ifstream f(m_filename.c_str());
    if (!f.good())
    {
      return false;
    }
  }

  return true;
}

void Task_LoadImg::task()
{
  if (!m_result.data)
  {
    m_result = cv::imread(m_filename, cv::IMREAD_ANYCOLOR);
  }

  while (!m_result.data && std::chrono::system_clock::now() < m_wait_images_until)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    m_result = cv::imread(m_filename, cv::IMREAD_ANYCOLOR);
  }

  if (!m_result.data)
  {
    throw std::runtime_error("Could not load " + m_filename);
  }

  // Store original image size and set valid area as the entire original image
  m_orig_size = m_result.size();
  m_valid_area = cv::Rect(0, 0, m_result.cols, m_result.rows);
  
  // Store the original image before any padding
  m_original_image = m_result.clone(); 

  // Get wavelet decomposition levels required for the image
  cv::Size expanded;
  int levels = Task_Wavelet::levels_for_size(m_orig_size, &expanded);
  std::string name = basename();
  
  // Log original and expanded dimensions
  m_logger->verbose("%s has resolution %dx%d, using %d wavelet levels and expanding to %dx%d\n",
                    name.c_str(), m_orig_size.width, m_orig_size.height, levels,
                    expanded.width, expanded.height);

  // Apply padding if needed for wavelet transform
  if (expanded != m_orig_size)
  {
    int expand_x = expanded.width - m_orig_size.width;
    int expand_y = expanded.height - m_orig_size.height;
    cv::Mat tmp(expanded.height, expanded.width, m_result.type());

    // Add padding with reflection at the borders
    cv::copyMakeBorder(m_result, tmp,
                       expand_y / 2, expand_y - expand_y / 2,
                       expand_x / 2, expand_x - expand_x / 2,
                       cv::BORDER_REFLECT);

    // Update result with padded image
    m_result = tmp;
    
    // Update valid area to point to the original image within the padded one
    m_valid_area = cv::Rect(cv::Point(expand_x / 2, expand_y / 2), m_orig_size);
  }
}
