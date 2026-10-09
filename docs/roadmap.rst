Roadmap
=======

.. The layout is inspired by the Zenodo roadmap
.. https://about.zenodo.org/roadmap/

This page provides an overview of current and future development goals for LSAPy. We value feedback from the community and
encourage you to contribute to the discussion on these goals; you can find the list of roadmap elements in the issue tracker,
under the `roadmap`_ label.

.. _roadmap: https://github.com/baptistehamon/lsapy/issues?q=is%3Aissue+state%3Aopen+label%3Aroadmap


.. grid:: 3
    :gutter: 2

    .. DON'T MODIFY THE FOLLOWING THREE CARDS

    .. grid-item-card:: Current
        :text-align: center
        :class-card: sd-border-0 sd-bg-transparent
        :shadow: none

        What we work on now

    .. grid-item-card:: Near-term
        :text-align: center
        :class-card: sd-border-0
        :shadow: none

        | What we plan to do
        | *(0-12 months)*

    .. grid-item-card:: Future
        :text-align: center
        :class-card: sd-border-0
        :shadow: none

        | What we investigate
        | *(6+ months)*

    .. START OF ROADMAP ELEMENTS

    .. grid-item-card:: JOSS Paper
        :class-header: sd-bg-success sd-text-white

        **Science/Research**
        ^^^
        Publish a paper describing LSAPy in the Journal of Open Source Software (JOSS).

    .. grid-item-card:: LSA Frameworks
        :class-header: sd-bg-danger sd-text-white

        **Development**
        ^^^
        Implement different LSA frameworks, such as the `FAO FLE`_ and a fuzzy-logic framework.

    .. grid-item-card:: Land Use Models
        :class-header: sd-bg-danger sd-text-white

        **Development**
        ^^^
        Extend LSAPy scope to include land use optimization or allocation models.

    .. grid-item-card:: Improve Documentation
        :class-header: sd-bg-primary sd-text-white

        **Documentation/Website**
        ^^^
        Enhance the documentation with a proper introduction to LSAPy, user-guides, and examples to help users get started with the library.
        +++
        *Discuss here*: [#77](https://github.com/baptistehamon/lsapy/issues/77)


.. div:: sd-text-center sd-font-italic

    Last updated: October 9, 2026


Guidelines for maintainers
--------------------------

The roadmap is a living document that evolves over time. New elements can be added to the roadmap as the project progresses and new goals are identified.
However, the roadmap is only meant to provide an overview of long-term project goals, and not every idea or suggestion will be suitable for inclusion.
As a rule of thumb, to be included in the roadmap, a suggestion should be a significant improvement or addition to the project,
or should require a substantial amount of work to be implemented. Minor bug fixes or enhancements limited in scope should remain as issues.

A roadmap element can be identified in several ways:

- From discussions in the project's issue tracker.
- From feedback from the community, through other communication channels.
- From internal planning by the project maintainers.

Ideally, each roadmap element should have a dedicated issue, where the idea can be further discussed and refined.
This issue should include a clear description of the problem being addressed, the proposed solution, and any relevant context or background information.
A link to this issue should be included in the roadmap element, so that anyone interested can easily find more information and participate in the discussion.

Before adding a new element to the roadmap, its relevant timeframe and category should be determined.

Timeframes
^^^^^^^^^^

The timeframe indicates when the elements is expected to be worked on or completed. It can be one of the following:

- **Current**: Items that are currently being worked on.
- **Near-term**: Items that are planned to be worked on in the near future (0-12 months).
- **Future**: Items that are being investigated for potential future work (6+ months).

Typically, most new elements should be in the near-term or future categories, depending on their complexity and the resources required to implement them.

Categories
^^^^^^^^^^

The category indicates the type of work being done, such as development, documentation, or community engagement.
Each roadmap element should be assigned to a relevant category, to help readers understand the nature of the work being proposed.
Five categories are defined for the roadmap elements:

- **Documentation/Website**: Elements related to the creation or improvement of the project's website, documentation, tutorials, or other educational resources.
- **Development**: Elements related to the development of new features, improvements to existing functionality, or other technical/maintenance work.
- **Science/Research**: Elements related to scientific research, data analysis, or other work that contributes to the project's scientific goals.
- **Community Engagement**: Elements related to outreach, communication, or other efforts to engage with the project's user community.
- **Administration/Management**: Elements related to project management, governance, or other administrative tasks.

Each category has a specific color associated with it, to help visually distinguish between different types of work in the roadmap.
The colors are based on the semantic color names defined by the sphinx-design extension, and are as follows:

.. ideally, custom colors should be used for the categories
.. however, the sphinx-design extension does not yet allow to use custom colors, and overriding the default colors does not seem to work.
.. if this issue is resolved in the future, the following colors should be used for the categories:
.. website/documentation: Blue (#00a0b0)
.. development: Red (#cc2a36)
.. science/research: Green (#bcd42a)
.. community engagement: Yellow (#eb6841)
.. administration/management: Orange (#edc951)


========================= ====== =============
Category                  Color  Semantic name
========================= ====== =============
Documentation/Website     Blue   ``primary``
Development               Red    ``danger``
Science/Research          Green  ``success``
Community Engagement      Purple ``secondary``
Administration/Management Orange ``warning``
========================= ====== =============


Layout and formatting
^^^^^^^^^^^^^^^^^^^^^

An element is added using ``.. grid-item-card::`` directive from the `sphinx-design`_ extension as follows:

.. code-block:: rst

        .. grid-item-card:: Title
            :class-header: sd-bg-<category-semantic-color> sd-text-white

            **Category**
            ^^^
            Description of the element, outlining its purpose and significance.
            +++
            *Discuss here*: #N (link to the issue)

The ``.. grid-item-card::`` directive creates a card with a title and description, and the ``:class-header:``
option allows you to specify the color of the card header based on the category of the element.

This card should be placed in the appropriate section of the roadmap (Current, Near-term, or Future) based on its
timeframe and chronological order (oldest (top) to newest (bottom)). ``sphinx-design`` automatically wraps the cards
in the grid layout, so when the number of cards exceeds the number of columns, it will automatically create a new row and place
the next card in the first column. As a result, to insert a new card at the right place, you may need to create a new "empty" card
with the following directive:

.. code-block:: rst

        .. grid-item-card::
            :class-card: sd-border-0 sd-bg-transparent
            :shadow: none

.. _FAO FLE: https://www.fao.org/4/x5310e/x5310e00.htm
.. _sphinx-design: https://sphinx-design.readthedocs.io/en/latest/
