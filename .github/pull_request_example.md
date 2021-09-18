// Убедитесь что название PR соответствует шаблону:
// Task00 <Имя> <Фамилия> <Аффиляция>
// И проверьте что обе ветки PR (отправляемая из вашего форкнутого репозитория и та в которую вы отправляете PR) называются task00

// Впишите сюда (между pre и /pre тэгами) вывод тестирования на вашем компьютере:

<details><summary>Локальный вывод</summary><p>

<pre>
Number of OpenCL platforms: 1
Platform #1/1
        Platform name: Intel(R) CPU Runtime for OpenCL(TM) Applications
        Vendor name: Intel(R) Corporation
        Number of Devices: 1
        Device #1/1
                Device name: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
                Device type: 2
                Device memory: 8251219968 bytes
                Device is available
                Device version: OpenCL 2.1 (Build 0)
                Driver version: 18.1.0.0920
</pre>

</p></details>

// Затем создайте PR, должна начать выполняться автоматическиая сборка на Travis CI - рядом с коммитом в PR появится оранжевый шарик (сборка в процессе),
// который потом станет зеленой галкой (прошло успешно) или красным крестиком (что-то пошло не так).
// Затем откройте PR на редактирование чтобы добавить в описание (тоже между pre и /pre тэгами) вывод тестирования на Travis CI:
// Чтобы его найти - надо нажать на зеленую галочку или красный крестик рядом с вашим коммитов в рамках PR

<details><summary>Вывод Travis CI</summary><p>

<pre>
$ ./enumDevices
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: 
The command "./enumDevices" exited with 0.
</pre>

</p></details>
