// Убедитесь что название PR соответствует шаблону:
// Task0N <Имя> <Фамилия> <Аффиляция>
// И проверьте что обе ветки PR (отправляемая из вашего форкнутого репозитория и та в которую вы отправляете PR) называются одинаково - task0N

// Впишите сюда (между pre и /pre тэгами) вывод тестирования на вашем компьютере:

<details><summary>Локальный вывод</summary><p>

<pre>
Number of OpenCL platforms: 1
no GPU device
Data generated for n=100000000!
Log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <aplusb> was successfully vectorized (8)
Done.
Kernel average time: 0.101015+-0.00081907 s
GFlops: 0.989956
VRAM bandwidth: 11.0636 GB/s
Result data transfer time: 0.0386717+-0.00118461 s
VRAM -> RAM bandwidth: 9.63311 GB/s
</pre>

</p></details>

// Затем создайте PR, должна начать выполняться автоматическиая сборка на Travis CI - рядом с коммитом в PR появится оранжевый шарик (сборка в процессе),
// который потом станет зеленой галкой (прошло успешно) или красным крестиком (что-то пошло не так).
// Затем откройте PR на редактирование чтобы добавить в описание (тоже между pre и /pre тэгами) вывод тестирования на Travis CI:
// Чтобы его найти - надо нажать на зеленую галочку или красный крестик рядом с вашим коммитов в рамках PR.
// P.S. В случае если Travis CI сборка не запустилась - попробуйте через десять минут или через час добавить фиктивный коммит (например добавив где-то пробел).

<details><summary>Вывод Travis CI</summary><p>

<pre>
$ ./enumDevices
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: 
The command "./enumDevices" exited with 0.
</pre>

</p></details>
