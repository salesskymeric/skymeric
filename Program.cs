using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Timers;
using Python.Runtime; // Make sure you've added Python.Runtime.dll as a reference

class Program
{
    static string modelDir = "/home/user1/hfllama2/Llama-2-7b-chat-hf";
    static HashSet<string> processedFiles = new HashSet<string>();

    static void Main(string[] args)
    {
        PythonEngine.Initialize();
        using (Py.GIL())
        {
            dynamic transformers = Py.Import("transformers");
            dynamic torch = Py.Import("torch");

            string questionsFolderPath = "/home/user1/hfllama2/questions"; // Update with the path to your questions folder
            string answersFolderPath = "/home/user1/hfllama2/answers"; // Update with the path to your answers folder

            // Create the answers folder if it doesn't exist
            Directory.CreateDirectory(answersFolderPath);

            // Start a timer to periodically check for new files
            System.Timers.Timer timer = new System.Timers.Timer();
            timer.Interval = 5000; // Check every 5 seconds (adjust as needed)
            timer.Elapsed += (sender, e) => CheckForNewFiles(questionsFolderPath, answersFolderPath);
            timer.Start();

            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();

            // Stop the timer when the program is about to exit
            timer.Stop();
        }
    }

    static void CheckForNewFiles(string questionsFolderPath, string answersFolderPath)
    {
        string[] questionFiles = Directory.GetFiles(questionsFolderPath, "*.txt");

        foreach (string questionFile in questionFiles)
        {
            // Check if the question has already been processed
            if (processedFiles.Contains(questionFile))
                continue; // Skip if the file has already been processed

            // Process the new question
            ProcessQuestion(questionFile, answersFolderPath);

            // Add the file to the set of processed files
            processedFiles.Add(questionFile);
        }
    }

    static void ProcessQuestion(string questionFilePath, string answersFolderPath)
    {
        using (Py.GIL())
        {
            dynamic transformers = Py.Import("transformers");
            dynamic torch = Py.Import("torch");

            dynamic tokenizer = transformers.AutoTokenizer.from_pretrained(modelDir);
            dynamic model = transformers.AutoModelForCausalLM.from_pretrained(modelDir);

            tokenizer.pad_token = "[PAD]";
            tokenizer.padding_side = "left";

            string question = File.ReadAllText(questionFilePath);

            // Prepare your questions
            string[] questions = { question };

            // Tokenize the questions
            dynamic tokenizedQuestions = tokenizer(questions, return_tensors: "pt", padding: true, truncation: true);

            // Batch processing
            dynamic inputIds = tokenizedQuestions["input_ids"];
            dynamic attentionMask = tokenizedQuestions["attention_mask"];

            using (torch.no_grad())
            {
                try
                {
                    dynamic outputs = model.generate(input_ids: inputIds, attention_mask: attentionMask, max_length: 200, num_return_sequences: 1);

                    dynamic answers = new PyList();
                    foreach (PyObject output in outputs)
                    {
                        answers.Append(tokenizer.decode(output, skip_special_tokens: true));
                    }

                    // Write answers to corresponding files
                    string answerFilePath = Path.Combine(answersFolderPath, Path.GetFileNameWithoutExtension(questionFilePath) + "_answer.txt");

                    for (int i = 0; i < questions.Length; i++)
                    using (StreamWriter writer = File.AppendText(answerFilePath))
                    {
                        writer.WriteLine($"Question: {questions[i]}");
                        writer.WriteLine($"Answer: {answers[i]}");
                        writer.WriteLine();
                    }

                    Console.WriteLine($"Answer for question '{question}' has been written to '{answerFilePath}'");
                }
                catch (PythonException e)
                {
                    Console.WriteLine("Error during generation: " + e.Message);
                    // Print tokenized inputs for debugging
                    Console.WriteLine("Tokenized inputs: " + tokenizedQuestions);
                    throw e;
                }
            }
        }
    }
}
