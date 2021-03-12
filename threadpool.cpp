/* \file
   \brief A demo threadpool implementation
*/

#include "threadpool.hpp"

ThreadPool::ThreadPool(size_t thread_count)
  : task_count_(0u), stop_(false)
{
  for (size_t i = 0; i < thread_count; i++)
  {
    // Each thread executes this lambda
    workers_.emplace_back([this]()->void
                          {
                            while (true)
                            {
                              std::function<void()> task;
                              { // acquire lock
                                std::unique_lock<std::mutex> lock(mutex_);
                                condition_.wait(lock, [this]()->bool
                                                {
                                                  return !tasks_.empty() || stop_;
                                                });
                                if (stop_ && tasks_.empty())
                                {
                                  return;
                                }
                                task = std::move(tasks_.front());
                                tasks_.pop();
                              } // release lock
                              task();
                              task_count_--;
                            }
                          });
  }
}

ThreadPool::~ThreadPool()
{
  stop_ = true;
  condition_.notify_all();
  for (auto& w: workers_)
  {
    w.join();
  }
}

void ThreadPool::schedule(const std::function<void()>& task)
{
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.push(task);
  }
  task_count_++;
  condition_.notify_one();
}

void ThreadPool::waitAll() const
{
  while (task_count_ != 0u)
  {
    std::this_thread::yield();
  }
}
