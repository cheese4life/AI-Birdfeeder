# AI Disclosure

> **Project:** Smart Birdhouse  
> **Author:** Anton Bloch (antbloch@uw.edu)  
> **Date:** April 2026

---

## Purpose

This document describes how AI tools were used during development of this project.

## How AI Was Used

### Documentation (README, setup guides)

The `README.md` setup guide was written and validated by the author. AI (GitHub Copilot) was used to assist with formatting and to verify that the described steps matched the actual code behavior.

### Code & Software Architecture

All source code, system architecture, and feature design in `web.py` and `detector.py` was implemented by hand. Design decisions (pipeline structure, multi-threaded camera/motion/detection/upload architecture, frame differencing approach, S3 + Bedrock integration) were made by the author based on domain knowledge and project requirements.

### Engineering Best Practices

AI was consulted to **verify** choices around project structure, configuration patterns, `.gitignore` conventions, and credential handling. These were treated as a reference, not a source of truth.

### Debugging & Problem-Solving

AI was used as a diagnostic tool to work through blocking technical issues (environment configuration, ffmpeg pipeline tuning, thread synchronization) that would have otherwise delayed productivity.

AI was not used for the following:
- Generating entire source code or algorithms
- Making design decisions without author review
- Producing any output that was accepted without verification

## Tools

- **GitHub Copilot** (Claude Sonnet 4.6): conversational debugging, documentation review, best-practice verification

---

*This disclosure follows emerging best practices for AI transparency in academic and research software projects.*
