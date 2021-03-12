/*! \file
    \brief A minimalistic thread pool implementation.
    Code based on:
    https://codereview.stackexchange.com/questions/79323/simple-c-thread-pool
    https://github.com/vit-vit/CTPL/blob/master/ctpl_stl.h
*/

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>

class ThreadPool
{
public:
  explicit ThreadPool(size_t thread_count);
  ~ThreadPool();

  void schedule( const std::function<void()>&);

  void waitAll() const;

private:

  // Make the thread pool noncopyable
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool & operator=(const ThreadPool&) = delete;
  ThreadPool & operator=(const ThreadPool&&) = delete;
  
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::atomic_uint task_count_;
  std::atomic_bool stop_;

  std::mutex mutex_;
  std::condition_variable condition_;
};
