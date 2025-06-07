// Work queue that distributes alignment and merging operations between threads.

#pragma once
#include <thread>
#include <deque>
#include <vector>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <exception>
#include <opencv2/core/core.hpp>
#include "logger.hh"

namespace focusstack {

// Generic runnable task, optionally with some dependencies on other tasks
class Task
{
public:
  Task();
  virtual ~Task();

  virtual bool ready_to_run();
  virtual bool uses_opencl() { return false; }
  bool is_running() const { return m_running; }
  bool is_completed() const { return m_done; }
  void run(std::shared_ptr<Logger> logger = nullptr);
  std::string filename() const { return m_filename; }
  std::string name() const { return m_name; }
  std::string basename() const;
  int index() const { return m_index; }
  void set_index(int index) { m_index = index; }
  const std::vector<std::shared_ptr<Task> > &get_depends() const { return m_depends_on; }

  void wait();

protected:
  virtual void task() { };

  std::shared_ptr<Logger> m_logger;
  std::string m_filename;
  int m_index;
  int m_jpgquality;
  std::string m_name;
  std::mutex m_mutex;
  std::vector<std::shared_ptr<Task> > m_depends_on; // List of tasks this task needs as inputs

  std::condition_variable m_wakeup;
  bool m_running;
  bool m_done;
};

// Task that has image as a result.
class ImgTask: public Task
{
public:
  ImgTask() {};
  ImgTask(cv::Mat result): m_result(result) {}
  virtual const cv::Mat &img() const { return m_result; }

  bool has_valid_area() const { return m_valid_area.width != 0 && m_valid_area.height != 0; }
  cv::Rect valid_area() const {
    if (!has_valid_area())
    {
      if (m_logger) m_logger->verbose("Valid area not defined for %s, using default.\n", m_filename.c_str());
      return cv::Rect(0, 0, m_result.cols, m_result.rows);
    }
    else
    {
      return m_valid_area;
    }
  }

  cv::Mat img_cropped() const {
    cv::Mat result = img();
    
    // First extract the valid area (removing any padding)
    if (has_valid_area() && (m_valid_area.x > 0 || m_valid_area.y > 0 || 
        m_valid_area.width < result.cols || m_valid_area.height < result.rows)) {
      result = extract_original_area(result);
    }
    
    return result;
  }

  // Extract the original image area from an expanded image
  cv::Mat extract_original_area(const cv::Mat& expanded_img) const
  {
    if (!has_valid_area()) {
      return expanded_img; // No valid area defined, return as is
    }
    
    // Check if the image is already the right size or the valid area covers the whole image
    if (m_valid_area.x == 0 && m_valid_area.y == 0 && 
        m_valid_area.width == expanded_img.cols && m_valid_area.height == expanded_img.rows) {
      return expanded_img;
    }
    
    // Extract the valid area from the expanded image
    cv::Rect safeRect = m_valid_area;
    
    // Make sure the rectangle is within image bounds
    safeRect.x = std::max(0, std::min(safeRect.x, expanded_img.cols - 1));
    safeRect.y = std::max(0, std::min(safeRect.y, expanded_img.rows - 1));
    safeRect.width = std::min(safeRect.width, expanded_img.cols - safeRect.x);
    safeRect.height = std::min(safeRect.height, expanded_img.rows - safeRect.y);
    
    // Extract and return the valid region
    return expanded_img(safeRect).clone();
  }

protected:
  cv::Mat m_result;
  cv::Rect m_valid_area;

  // Limit valid area by intersection
  void limit_valid_area(cv::Rect other)
  {
    int x0 = std::max(m_valid_area.x, other.x);
    int y0 = std::max(m_valid_area.y, other.y);
    int x1 = std::min(m_valid_area.x + m_valid_area.width, other.x + other.width);
    int y1 = std::min(m_valid_area.y + m_valid_area.height, other.y + other.height);
    m_valid_area = cv::Rect(x0, y0, x1 - x0, y1 - y0);
  }
};

// Work queue class that distributes tasks to threads.
class Worker
{
public:
  Worker(int max_threads, std::shared_ptr<Logger> logger);
  ~Worker();

  // Add task to the end of the queue
  void add(std::shared_ptr<Task> task);

  // Prepend task, causing it to run as soon as possible
  void prepend(std::shared_ptr<Task> task);

  // Wait until all tasks have finished
  bool wait_all(int timeout_ms = -1);

  bool failed() const { return m_failed; }

  std::string error() const { return m_error; }

  void get_status(int &total_tasks, int &completed_tasks, std::string &running_task_name);

private:
  std::shared_ptr<Logger> m_logger;
  std::vector<std::thread> m_threads;
  std::deque<std::shared_ptr<Task> > m_tasks;
  std::unordered_set<std::shared_ptr<Task> > m_running;

  bool m_closed;
  int m_tasks_started;
  int m_total_tasks;
  int m_completed_tasks;
  int m_opencl_users;
  int m_wait_count;

  bool m_failed;
  std::string m_error;

  std::mutex m_mutex;
  std::condition_variable m_wakeup;

  std::chrono::time_point<std::chrono::steady_clock> m_start_time;
  float seconds_passed() const;

  void worker(int thread_idx);
};


}
