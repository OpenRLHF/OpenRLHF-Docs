Welcome to OpenRLHF's documentation!
===================================

`OpenRLHF <https://github.com/OpenRLHF/OpenRLHF>`_ is an easy-to-use, scalable and high-performance RLHF Framework built on Ray, DeepSpeed and HF Transformers.

- **Simple and Easy-to-use**: OpenRLHF stands out as one of the simplest high-performance RLHF libraries available, fully compatible with Huggingface models and datasets.
- **High Performance**: In RLHF training, 80% of the time is spent on the sample generation stage. OpenRLHF excels here, leveraging large inference batch sizes with Ray, Adam Offload (Pinned Memory), and vLLM generation acceleration. 
- **Distributed RLHF**: OpenRLHF distributes the Actor, Reward, Reference, and Critic models across separate GPUs using Ray, while placing the Adam optimizer on the CPU. This setup enables full-scale fine-tuning of models with 70B+ parameters on multiple A100 80G GPUs and vLLM, as well as 7B models across multiple 24GB RTX 4090 GPUs。
- **PPO Implementation Optimization**: To enhance training stability, we have integrated various implementation tricks for PPO, details in `Zhihu <https://zhuanlan.zhihu.com/p/622134699>`_ and  the `Notion AI blog <https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f>`_.

For more technical details, see our `technical report <https://arxiv.org/abs/2405.11143>`_ and `slides <https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing>`_.

Check out the :doc:`quick_start` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   quick_start
   usage
   performance
   multi-node
   remote_rm
   checkpoint
