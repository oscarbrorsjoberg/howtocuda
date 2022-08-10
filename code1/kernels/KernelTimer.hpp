class KernelTimer
{

 public:
  KernelTimer();
  ~KernelTimer();

	void start();
	void stop();

 private:
	cudaEvent_t start_;
	cudaEvent_t stop_;

};
